# Software Name : s3lspeech
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.

import os 
import logging
import datasets
from typing_extensions  import override

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.models.decoder import download_pretrained_files
from torchaudio.models.decoder import ctc_decoder

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from config.s3lspeech import S3LSpeechConfig
from config.block import BlockConfig

from modules.train import (
    GreedyCTCDecoder,
    TrainRequeue
)
from modules.external import (
    ConvFeatureExtractionModel, 
    ProjUpsampling
)
from modules.perturb import SpecAugment
from models.ssl import (
    S3LSpeechDT,
    S3LSpeech
)
from modules.simple_wer import SimpleWER

class S3LSpeechASR(S3LSpeechDT):
    """
    Subclass of S3LSpeechDT with finetuning for automatic speech recognition
    """
    def __init__(self, config):
        super().__init__(config)
        if self.pretraining == True:
            logging.info('Loading pretrained model {}...'.format(
                self.p_checkpoint))
            self.essl = S3LSpeech.load_from_checkpoint(
                self.p_checkpoint, config=self.config)
        else:
            self.essl = S3LSpeech(config=self.config)
        
        self.val_beam_wers = SimpleWER()
        self.val_greed_wers = SimpleWER()
        self.val_step_wers = []

        # Decoders
        self.load_ctc_decoder()
        self.greedy_decoder = GreedyCTCDecoder(self.tokens)
        self.build_classifier_asr()

    def load_ctc_decoder(self):
        """
        Load CTC decoder using librispeech 4-gram language model

        https://pytorch.org/audio/stable/tutorials/asr_inference_with_ctc_decoder_tutorial.html
        """
        torch.hub.set_dir(self.data_dir)
        lm_files = download_pretrained_files("librispeech-4-gram")
        self.tokens = []
        self.token_class = {}
        with open(lm_files.tokens) as tokens_file:
            lines = tokens_file.readlines()
            num_tokens = 0
            for line in lines:
                line = line.strip()
                self.tokens.append(line)
                self.token_class[line] = num_tokens
                num_tokens += 1
        logging.info("Tokens %s" % self.tokens)
        logging.info("Token class %s" % self.token_class)

        self.beam_search_decoder = ctc_decoder(
            lexicon=lm_files.lexicon,
            tokens=lm_files.tokens,
            lm=lm_files.lm,
            nbest=3,
            beam_size=self.beam_size,
            lm_weight=self.lm_weight,
            word_score=self.word_score,
            )

    def build_classifier_asr(self):
        classifier_asr_cfg = BlockConfig()
        self.avg_weights = nn.Parameter(torch.ones( 
            size=(classifier_asr_cfg.encoder_layers - 2, 1, 1, 1)))
        nn.init.xavier_normal_(self.avg_weights)

        classifier_asr_cfg.conv_feature_layers = \
            "[(512, 5, 1), (512, 5, 1)]"
        classifier_asr_cfg.extractor_mode = "default"
        classifier_asr_cfg.normalize = False
        classifier_asr_cfg.conv_bias = 0.0
        self.classifier_asr = nn.Sequential(
            ProjUpsampling(
                in_channels=classifier_asr_cfg.encoder_embed_dim,
                rate=8, 
                filters=512, 
                kernel_size=(5,),
                norm_type='ln', 
                act_func='relu', 
                dropout=classifier_asr_cfg.dropout),
            ConvFeatureExtractionModel(
                conv_layers=eval(classifier_asr_cfg.conv_feature_layers),
                dropout=classifier_asr_cfg.dropout,
                input_d=512,
                mode=classifier_asr_cfg.extractor_mode,
                conv_bias=classifier_asr_cfg.conv_bias,
                ),
            nn.Linear(
                in_features=512, out_features=len(self.tokens)),
            nn.ReLU(),
            )

    def _step(self, x_student, words):
        targets = []
        target_lengths = []
        for i in range(len(words)):
            sentence = words[i]
            target_lengths.append(len(sentence))
            for char in sentence:
                targets.append(self.token_class.get(char,self.token_class['|']))

        y_student = self.essl.student(x_student)

        # Weighted sum of layers
        layers  = [self.essl.student[3].layer_features[i] for i in range(
            BlockConfig().encoder_layers - 2)]
        layers_wmean = (self.avg_weights * torch.stack(layers)).sum(
            dim=0) / self.avg_weights.sum()
        y_student = layers_wmean
        
        y_student = self.classifier_asr(y_student)

        T = y_student.shape[1] # Sequence length 
        N = y_student.shape[0] # Batch size
        targets = torch.as_tensor(
            targets, dtype=torch.long, device=y_student.device)
        target_lengths = torch.as_tensor(
            target_lengths, dtype=torch.long, device=y_student.device)
        input_lengths = torch.full(
            size=(N,), fill_value=T, dtype=torch.long, device=y_student.device)
        
        # Convert from B x T x C to T x B x C
        y_student = y_student.transpose(0, 1)
        ctc_loss = torch.nn.CTCLoss(
            blank=0, reduction='none', zero_infinity=False)
        log_softmax = torch.nn.LogSoftmax(dim=2)
        log_probs = log_softmax(y_student)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss = torch.mean(loss)
        # Convert from T x B x C to B x T x C
        y_student = y_student.transpose(0, 1).float()

        return loss, y_student

    def finetune_model(self, batch, batch_idx):
        #https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
        #https://pytorch.org/audio/stable/tutorials/asr_inference_with_ctc_decoder_tutorial.html
        #https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        #https://distill.pub/2017/ctc/
        signal  = batch.signal.data
        words   = batch.words # List with sentences for audio sequences

        if self.freeze_finetuning == True:
            logging.info("Freezing backbone model ...")
            for params in [self.essl.student.parameters(), 
                self.essl.predictor.parameters(), 
                self.essl.teacher.parameters()]:
                for param in params:
                    param.requires_grad = False

        x_student = signal
        x_student = x_student.transpose(1, 2) # BxTxC to BxCxT
        B, C, T = x_student.shape
        x_student_lengths = torch.full(
            size=(B,), fill_value=T, dtype=torch.long, device=x_student.device)
        self.specaug = SpecAugment(
            freq_masks=int(0.020 * C), freq_width=20,
            time_masks=int(0.025 * T), time_width=20,
            max_time_masks=100, gauss_mask_std=1.0)
        x_student = self.specaug(x_student, x_student_lengths)
        x_student = x_student.transpose(1, 2) # BxCxT to BxTxC
        loss, y_student = self._step(x_student, words)

        logging.info(torch.abs(self.avg_weights).view(-1,))
        logging.info(("Fine tuning, pytorch loss = {0}" +
            " step = {1} batch id = {2} epoch = {3} lr = {4}").format(
            loss, self.global_step, batch_idx, self.current_epoch, 
            self.optimizers().param_groups[0]['lr']))
        self.log('ft_loss', loss)
        self.log('ft_lr', self.optimizers().param_groups[0]['lr']) 
        return loss

    def shared_eval_step(self, batch, batch_idx, test_type):
        signal  = batch.signal.data
        words   = batch.words # List with sentences for audio sequences

        wer = SimpleWER() 
        x_student = signal
        loss, y_student = self._step(x_student, words)

        logging.info((test_type + ", " +
            "batch_loss = {0} step = {1} batch id = {2} epoch = {3}").format(
            loss, self.global_step, batch_idx, self.current_epoch))
        
        # https://pytorch.org/audio/stable/tutorials/asr_inference_with_ctc_decoder_tutorial.html
        log_softmax = torch.nn.LogSoftmax(dim=2)
        emissions = log_softmax(y_student)
        emissions = emissions.cpu()

        for i in range(len(words)):
            actual_transcript = words[i]
            emission = emissions[i]
            emission = emission.unsqueeze(0)

            greedy_result = self.greedy_decoder(emission[0])
            greedy_transcript = " ".join(greedy_result).strip()
            self.val_greed_wers.AddHypRef(
                greedy_transcript, actual_transcript)
           
            beam_search_result = self.beam_search_decoder(emission)
            beam_search_transcript = ""
            if len(beam_search_result) > 0 and len(beam_search_result[0]) > 0:
                beam_search_transcript = " ".join(
                    beam_search_result[0][0].words).strip()
            self.val_beam_wers.AddHypRef(
                beam_search_transcript, actual_transcript)
            wer.AddHypRef(beam_search_transcript, actual_transcript)

        self.val_step_losses.append(loss)
        self.val_step_wers.append(wer.GetWER())
        return loss
    
    def shared_eval_epoch_end(self, test_type):
        loss = sum(self.val_step_losses) / len(self.val_step_losses)
        greedy = self.val_greed_wers.GetWER()
        beam = self.val_beam_wers.GetWER()
        logging.info((test_type + ", " +
            "loss = {0} greedy = {1} beam = {2} step = {3} epoch = {4}").format(
            loss, greedy, beam, self.global_step, self.current_epoch))
        logging.info((test_type + ", " +
            "wers list = {0}").format(self.val_step_wers))
        
        self.val_step_losses.clear()  # free memory
        self.val_step_wers.clear()
        self.val_beam_wers = SimpleWER()
        self.val_greed_wers = SimpleWER()

    def finetune_model_requeue(
        self, pretrain_dataloader, finetune_dataloader, val_dataloader):
        """
        Use Pytorch Lightning to train the model, requeuing the trianing job

        Args
            pretrain_dataloader: data for self-supervised pretraining
            finetune_dataloader: data for supervised finetuning
            val_dataloader: data for validating the module
        Returns:

        [1] https://pytorch-lightning.readthedocs.io/en/1.1.0/weights_loading.html#weights-loading
        [2] https://marcel.pages.gitlab.tech.orange/guidelines/requeue_short_jobs/
        [3] https://pytorch-lightning.readthedocs.io/en/1.1.0/trainer.html
        """
        logger = TensorBoardLogger('lightning_logs') 
        trainer = pl.Trainer( 
            devices=self.gpus, accelerator="gpu", precision=self.precision,
            max_steps=self.finetune_steps, logger=logger,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            accumulate_grad_batches=self.accumulate_grad_batches,
            callbacks=[TrainRequeue(
                nb_epochs_before_requeue=self.nb_epochs_before_requeue,
                checkpoint=self.checkpoint)])
        
        if os.path.exists(self.checkpoint):
            logging.info('Loading checkpoint {}...'.format(self.checkpoint))
            self = S3LSpeechASR.load_from_checkpoint(
                self.checkpoint, config=self.config)
            ckpt_path = self.checkpoint
        else:
            ckpt_path = None
            
        logging.info('Finetuning model ...')
        self.steps_per_epoch = len(finetune_dataloader)
        self.configure_optimizers = self.finetune_optimizers
        self.training_step = self.finetune_model
        trainer.fit(self, 
            train_dataloaders=finetune_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=ckpt_path)
        trainer.save_checkpoint(self.checkpoint)
    
    def finetune_librilight(
        self, pretrain_dataloader, finetune_dataloader, val_dataloader):
        """
        Use Pytorch Lightning to fine tune the model

        Args
            pretrain_dataloader: data for self-supervised pretraining
            finetune_dataloader: data for supervised finetuning
            val_dataloader: data for validating the module
        Returns:
        [1] https://pytorch-lightning.readthedocs.io/en/1.1.0/trainer.html
        """
        if self.pretraining == True:
            logging.info('Loading pretrained model {}...'.format(
                self.p_checkpoint))
            self = self.load_from_checkpoint(
                self.p_checkpoint, config=self.config)

        logging.info('Finetuning model ...')
        self.steps_per_epoch = len(finetune_dataloader)
        self.configure_optimizers = self.finetune_optimizers
        self.training_step = self.finetune_model
        logger = TensorBoardLogger('lightning_logs') 
        trainer = pl.Trainer(
            devices=self.gpus, accelerator="gpu", precision=self.precision,
            max_steps=self.finetune_steps, logger=logger,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            accumulate_grad_batches=self.accumulate_grad_batches)
        trainer.fit(self, 
            train_dataloader=finetune_dataloader,
            val_dataloaders=val_dataloader)
        trainer.save_checkpoint(self.checkpoint)

    def test_model(self, test_dataloader):
        """
        Use Pytorch Lightning to load model from checkpoint and test it
        Args
            test_dataloader: data for testing the module. It can be a list of
            datasets
        Returns:

        [1] https://pytorch-lightning.readthedocs.io/en/1.1.0/introduction_guide.html
        [2] https://pytorch-lightning.readthedocs.io/en/1.1.0/trainer.html
        """
        self = S3LSpeechASR.load_from_checkpoint(
            self.checkpoint, config=self.config)
        trainer = pl.Trainer(
            devices=self.gpus, accelerator="gpu", precision=self.precision) 
        trainer.test(
            self, dataloaders=test_dataloader, ckpt_path=self.checkpoint)
    
    def grid_search_beam(self, test_dataloader):
        """
        Use Pytorch Lightning to load model from checkpoint and test it, 
        changing the beam search decoder parameters
        Args
            test_dataloader: data for testing the module. It can be a list of
            datasets
        Returns:
        [1] https://pytorch-lightning.readthedocs.io/en/1.1.0/introduction_guide.html
        """
                
        beam_sizes = [500, 1000]#, 1500]
        lm_weights = [2.46, 3.23, 5.0, 7.0]
        word_scores = [-0.1, -0.26, -0.4]
        for beam_size in beam_sizes:
            for lm_weight in lm_weights:
                for word_score in word_scores:
                    self = S3LSpeechASR.load_from_checkpoint(
                        self.checkpoint, config=self.config)
                    self.beam_size = beam_size
                    self.lm_weight = lm_weight
                    self.word_score = word_score
                    self.load_ctc_decoder()
                    logging.info(('Beam ctc decoder params: beam size {0} ' +
                        'lm weight {1} word score {2}').format(
                        self.beam_size, self.lm_weight, self.word_score))
                    trainer = pl.Trainer(
                        devices=self.gpus, accelerator="gpu", 
                        precision=self.precision,)
                    trainer.test(self, dataloaders=test_dataloader, 
                        ckpt_path=self.checkpoint)
                    
def efficient_grid_search():
    config = S3LSpeechConfig()
    librispeech_dev_other = datasets.load_librispeech(
        data_root=config.data_dir, data_sets='dev_other',
        max_batch_length=config.batch_length)
    librispeech_dev_clean = datasets.load_librispeech(
        data_root=config.data_dir, data_sets='dev_clean',
        max_batch_length=config.batch_length)
    config = S3LSpeechConfig()
    config.checkpoint = config.checkpoint.replace('0.ckpt', 'ls100hr.ckpt')

    logging.info('Testing with LibriSpeech dev other ...')
    model = S3LSpeechASR(config)
    model.grid_search_beam(test_dataloader=librispeech_dev_other)

    logging.info('Testing with LibriSpeech dev clean ...')
    model = S3LSpeechASR(config)
    model.grid_search_beam(test_dataloader=librispeech_dev_clean)

def efficient_test(config):
    librispeech_dev_other = datasets.load_librispeech(
        data_root=config.data_dir, data_sets='dev_other',
        max_batch_length=config.batch_length)
    librispeech_dev_clean = datasets.load_librispeech(
        data_root=config.data_dir, data_sets='dev_clean',
        max_batch_length=config.batch_length)
    librispeech_test_other = datasets.load_librispeech(
        data_root=config.data_dir, data_sets='test_other',
        max_batch_length=config.batch_length)
    librispeech_test_clean = datasets.load_librispeech(
        data_root=config.data_dir, data_sets='test_clean',
        max_batch_length=config.batch_length)
    
    logging.info('Testing with LibriSpeech dev other ...')
    model = S3LSpeechASR(config)
    model.test_model(test_dataloader=librispeech_dev_other)
    logging.info('Testing with LibriSpeech dev clean ...')
    model = S3LSpeechASR(config)
    model.test_model(test_dataloader=librispeech_dev_clean)
    logging.info('Testing with LibriSpeech test other ...')
    model = S3LSpeechASR(config)
    model.test_model(test_dataloader=librispeech_test_other)
    logging.info('Testing with LibriSpeech test clean ...')
    model = S3LSpeechASR(config)
    model.test_model(test_dataloader=librispeech_test_clean)
    
def asr_finetune_requeue(dataset="ls100hr"):
    """
    Finetune EfficientASR model.

    dataset: ls100hr Librispeech train clean 100hr
             ll10hr  Librilight 10hr
             ll1hr   Librilight 1hr
             ll10min Librilight 10min
    """
    config = S3LSpeechConfig()
    librispeech_train_all = datasets.load_librispeech(
        data_root=config.data_dir, data_sets='all_train',
        max_batch_length=config.batch_length)
    librispeech_dev_other = datasets.load_librispeech(
        data_root=config.data_dir, data_sets='dev_other',
        max_batch_length=config.batch_length)
    
    config = S3LSpeechConfig()
    if dataset == "ls100hr":
        dataloader = datasets.load_librispeech(
            data_root=config.data_dir, data_sets='train_clean_100',
            max_batch_length=config.batch_length)
        config.checkpoint = config.checkpoint.replace('0.ckpt', 'ls100hr.ckpt')
        experiment = '\n\n--Experiment with train_clean_100 config {0}'.format(
            config.__dict__)
    elif dataset == "ll10hr":
        dataloader = datasets.load_librilight(
            data_root=config.data_dir, data_sets='librilight10h',
            max_batch_length=config.batch_length)
        config.finetune_steps = 80000
        config.finetune_lr = 3e-6
        config.batch_length = 18 * 60
        config.accumulate_grad_batches = 1
        config.checkpoint = config.checkpoint.replace('0.ckpt', 'll10hr.ckpt')
        experiment = '\n\n--Experiment with librilight 10hr config {0}'.format(
        config.__dict__)
    elif dataset == "ll1hr":
        dataloader = datasets.load_librilight(
            data_root=config.data_dir, data_sets='librilight1h',
            max_batch_length=config.batch_length)
        config.finetune_steps = 20000
        config.finetune_lr = 3e-6
        config.batch_length = 6 * 60
        config.accumulate_grad_batches = 1
        config.checkpoint = config.checkpoint.replace('0.ckpt', 'll1hr.ckpt')
        experiment = '\n\n--Experiment with librilight 1hr config {0}'.format(
        config.__dict__)
    elif dataset == "ll10min":
        dataloader = datasets.load_librilight(
            data_root=config.data_dir, data_sets='librilight10min',
            max_batch_length=config.batch_length)
        config.finetune_steps = 10000
        config.finetune_lr = 3e-6
        config.batch_length = 1 * 60
        config.accumulate_grad_batches = 1
        config.checkpoint = config.checkpoint.replace('0.ckpt', 'll10min.ckpt')
        experiment = '\n\n--Experiment with librilight 10min config {0}'.format(
        config.__dict__)
        
    model = S3LSpeechASR(config)
    logging.info(experiment)
    model.finetune_model_requeue(
        pretrain_dataloader=librispeech_train_all,
        finetune_dataloader=dataloader,
        val_dataloader=librispeech_dev_other)
    efficient_test(config)

def asr_finetune_librilight():
    asr_finetune_requeue(dataset="ll10hr")
    asr_finetune_requeue(dataset="ll1hr")
    asr_finetune_requeue(dataset="ll10min")