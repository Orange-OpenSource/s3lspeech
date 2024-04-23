# Software Name : s3lspeech
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.

import logging, os
import math
from typing_extensions  import override

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import datasets
from config.s3lspeech import S3LSpeechConfig
from modules.train import (
    TriStageLRScheduler,
    TrainRequeue
)
from modules.external import (
    ConvFeatureExtractionModel, 
    TransformerEncoder,
)
from modules.perturb import (
    SpecAugment,
    RandomShift,
)
from config.block import (
    BlockConfig,
    ShiftPerturbConfig
) 

class S3LSpeech(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config.data_dir
        self.checkpoint = os.path.join(config.data_dir, config.checkpoint)
        self.p_checkpoint = os.path.join(config.data_dir, config.p_checkpoint)
        self.freeze_finetuning = config.freeze_finetuning
        self.pretraining = config.pretraining
        self.precision = config.precision
        self.gpus = config.gpus

        self.nb_epochs_before_requeue = config.nb_epochs_before_requeue
        self.pretrain_steps = config.pretrain_steps
        self.finetune_steps = config.finetune_steps
        self.pretrain_lr = config.pretrain_lr
        self.finetune_lr = config.finetune_lr
        self.check_val_every_n_epoch = config.check_val_every_n_epoch
        self.accumulate_grad_batches = config.accumulate_grad_batches 
        self.optimizer_betas = config.optimizer_betas
        self.optimizer_weight_decay = config.optimizer_weight_decay

        self.n_negatives = config.n_negatives 
        self.cross_sample_negatives = config.cross_sample_negatives
        self.logit_temp = config.logit_temp
        
        self.beam_size = config.beam_size
        self.lm_weight = config.lm_weight
        self.word_score = config.word_score

        # Model
        logging.info('Building model ...')
        self.student = self.create_model(is_student=True)
        self.teacher = self.create_model(is_student=False)
        logging.info('Student \n{0}'.format(self.student))
        logging.info('Teacher \n{0}'.format(self.teacher))

        self.former_step = -1
        self.val_step_losses = []

        # Augmentation 
        self.load_random_shift()

    def create_model(self, is_student: bool) -> torch.nn.Sequential:
        """
        Build the model for student or teacher modules, using Spiral Base 
        architecture as the template

        [1] https://openreview.net/forum?id=TBpg4PnXhYH

        Args:
            is_student: wether to add a convolutional predictor on top 
            of the module, for student only
        Returns:
            nn.Sequential with the components of the model     
        """
        conv1_cfg = BlockConfig()
        conv1_cfg.conv_feature_layers = "[(384, 5, 2), (512, 5, 2),(512, 1, 1)]"
        conv1_cfg.extractor_mode = "layer_norm"
        conv1_cfg.normalize = True
        conv1_cfg.mel_filters = self.config.n_mels
        conv1_cfg.dropout = 0.0
        conv1 = ConvFeatureExtractionModel(
            conv_layers=eval(conv1_cfg.conv_feature_layers),
            dropout=conv1_cfg.dropout,
            input_d= conv1_cfg.mel_filters,
            mode=conv1_cfg.extractor_mode,
            conv_bias=conv1_cfg.conv_bias,
        )

        transf1_cfg = BlockConfig()
        transf1_cfg.encoder_layers = 2
        transf1_cfg.extract_layer_features = is_student
        transf1 = TransformerEncoder(transf1_cfg)

        conv2_cfg = BlockConfig()
        conv2_cfg.conv_feature_layers ="[(1536, 5, 2), (512, 1, 1)]"
        conv2_cfg.extractor_mode = "layer_norm"
        conv2_cfg.normalize = True
        conv2_cfg.dropout = 0.0
        conv2 = ConvFeatureExtractionModel(
            conv_layers=eval(conv2_cfg.conv_feature_layers),
            dropout=conv2_cfg.dropout,
            input_d=transf1_cfg.encoder_embed_dim,
            mode=conv2_cfg.extractor_mode,
            conv_bias=conv2_cfg.conv_bias,
        )

        transf2_cfg = BlockConfig()
        transf2_cfg.encoder_layers -= 2 
        transf2_cfg.extract_layer_features = is_student
        transf2 = TransformerEncoder(transf2_cfg)

        projection_head = nn.Linear(
            in_features=transf2_cfg.encoder_embed_dim, out_features=256)

        model = nn.Sequential(
            conv1, transf1, conv2, transf2, projection_head)
        
        if is_student == True:
            predictor_cfg = BlockConfig()
            predictor_cfg.conv_feature_layers = \
                "[(256, 5, 1), (256, 5, 1), (256, 1, 1)]"
            predictor_cfg.extractor_mode = "layer_norm"
            predictor_cfg.normalize = True
            predictor_cfg.conv_padding = "same"
            predictor_cfg.conv_bias = 0.0
            self.predictor = ConvFeatureExtractionModel(
                conv_layers=eval(predictor_cfg.conv_feature_layers),
                dropout=predictor_cfg.dropout,
                input_d=256,
                mode=predictor_cfg.extractor_mode,
                padding=predictor_cfg.conv_padding,
                conv_bias=predictor_cfg.conv_bias,
                )
            
        return model
    
    def load_random_shift(self):
        """
        Add positional random shift to avoid the student to learn positional 
        from input poistional information 

        [1] https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/examples/asr/conf/spiral/spiral_base_pretrain_ls960_noise.py
        """
        shift_config = ShiftPerturbConfig(
            dist='uniform',
            shift_prob=1.0,
            max_ratio=0.5,
            unit=8,
            max=16,
            min=0,
            truncate=False)
        self.random_shift = RandomShift(shift_config)

    def forward(self, x):
        x_student = x
        y_student = self.student(x_student)
        y_teacher = self.teacher(x_student)
        y_student = self.predictor(y_student)
        return y_student

    def validation_step(self, batch, batch_idx):
        self.shared_eval_step(batch, batch_idx, "Validation")

    def test_step(self, batch, batch_idx):
        self.shared_eval_step(batch, batch_idx, "Test")

    def on_validation_epoch_end(self):
        self.shared_eval_epoch_end("Validation")

    def on_test_epoch_end(self):
        self.shared_eval_epoch_end("Test")

    def shared_eval_step(self, batch, batch_idx, test_type):
        signal  = batch.signal.data
        signal_noise  = batch.signal_noise.data

        x_teacher = signal
        x_student = signal_noise
        y_student = self.student(x_student)
        y_student = self.predictor(y_student)
        y_teacher = self.teacher(x_teacher)

        # Calculate contrastive loss
        T = y_teacher.shape[1] # Sequence length 
        N = y_teacher.shape[0] # Batch size
        y_teacher_lengths = torch.full(
            size=(N,), fill_value=T, dtype=torch.long, device=y_teacher.device)
        y_teacher = y_teacher.contiguous()
        y_student = y_student.contiguous()
        y_teacher = y_teacher.view(1, -1, y_teacher.size(-1))
        y_student = y_student.view(1, -1, y_student.size(-1))
        sampled_negatives, _ = self.sample_negatives_flat(
            y_teacher, y_teacher_lengths.tolist())
        loss, accuracy = self.contrastive_loss(
            y_student, y_teacher, sampled_negatives)
        
        logging.info(("batch_loss = {0} step = {1} " +
            "batch id = {2} epoch = {3} lr = {4} accuracy = {5}").format(
            loss, self.global_step, batch_idx, self.current_epoch, 
            self.optimizers().param_groups[0]['lr'], accuracy)) 
        self.val_step_losses.append(loss)
        return loss

    def shared_eval_epoch_end(self, test_type):
        loss = sum(self.val_step_losses) / len(self.val_step_losses)
        logging.info((test_type + ", " +
            "loss = {0} step = {1} epoch = {2}").format(
            loss,  self.global_step, self.current_epoch))
        self.log('pt_val_loss', loss)
        self.val_step_losses.clear()  # free memory

    def pretrain_model(self, batch, batch_idx):
        """
        https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/nemo/collections/asr/models/st2vec/st2vec_model.py
        """
        signal  = batch.signal.data
        signal_noise  = batch.signal_noise.data

        x_teacher = signal
        x_student = signal_noise
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
        y_student = self.student(x_student)
        y_student = self.predictor(y_student)

        B, T, C = x_teacher.shape
        x_teacher_lengths = torch.full(
            size=(B,), fill_value=T, 
            dtype=torch.long, device=x_teacher.device)
        x_teacher, x_teacher_lengths, teacher_shift_num, _, \
            teacher_r_shift_num = \
            self.random_shift.shift(x_teacher, x_teacher_lengths, 0.0)

        # Update teacher from student weights and calculate targets, skipping 
        # inner steps when using grad accumulation
        if self.global_step == 0:
            target_momentum = 0
            self.ema_update(self.teacher, self.student, target_momentum)
        elif self.global_step != self.former_step: 
            target_momentum = self.momentum_schedule(
                base_value=0.995, final_value=1.0, 
                max_steps=self.pretrain_steps, step=self.global_step, 
                type='cosine')
            self.ema_update(self.teacher, self.student, target_momentum)
            self.former_step = self.global_step
        with torch.no_grad():
            y_teacher = self.teacher(x_teacher)

        # Readjust tensors to account for random shifting
        B, T, C = y_teacher.shape
        y_teacher_lengths = torch.full(
            size=(B,), fill_value=T, dtype=torch.long, device=y_teacher.device)
        if teacher_shift_num > 0:
            y_teacher = y_teacher[:, teacher_shift_num:]
            y_teacher_lengths = y_teacher_lengths - teacher_shift_num
        else:
            assert teacher_shift_num == 0
        if teacher_r_shift_num > 0:
            y_teacher = y_teacher[:, :-teacher_r_shift_num]
            y_teacher_lengths = y_teacher_lengths - teacher_r_shift_num
        else:
            assert teacher_r_shift_num == 0

        assert y_student.shape[1] == y_teacher.shape[1]
        
        # Calculate contrastive loss
        T = y_teacher.shape[1] # Sequence length 
        N = y_teacher.shape[0] # Batch size
        y_teacher_lengths = torch.full(
            size=(N,), fill_value=T, dtype=torch.long, device=y_teacher.device)
        y_teacher = y_teacher.contiguous()
        y_student = y_student.contiguous()
        y_teacher = y_teacher.view(1, -1, y_teacher.size(-1))
        y_student = y_student.view(1, -1, y_student.size(-1))
        sampled_negatives, _ = self.sample_negatives_flat(
            y_teacher, y_teacher_lengths.tolist())
        loss, accuracy = self.contrastive_loss(
            y_student, y_teacher, sampled_negatives)
        
        logging.info(("Pretraining, pytorch loss = {0} step = {1} " +
            "batch id = {2} epoch = {3} lr = {4} accuracy = {5}").format(
            loss, self.global_step, batch_idx, self.current_epoch, 
            self.optimizers().param_groups[0]['lr'], accuracy))
        self.log('pt_loss', loss)
        self.log('pt_acc',  accuracy)
        self.log('pt_lr', self.optimizers().param_groups[0]['lr'])
        return loss
    
    def pretrain_optimizers(self):
        """
        [1] https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        [2] https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW
        [3] https://github.com/Lightning-AI/lightning/issues/3051
        """
        logging.info("Configuring pretrain optimizers ...")
        adam = torch.optim.AdamW(
            self.parameters(), 
            lr=self.pretrain_lr, 
            betas=self.optimizer_betas, 
            weight_decay=self.optimizer_weight_decay)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer=adam, max_lr=self.pretrain_lr,
                total_steps=self.pretrain_steps,
                pct_start=0.08, anneal_strategy='cos',
                div_factor=1e6, final_div_factor=1,
                ),
            'interval': 'step',
            'frequency': 1
            }
        return {'optimizer' : adam, 'lr_scheduler' : scheduler}

    def pretrain_model_requeue(
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
        if self.pretraining == False:
            return
        
        logger = TensorBoardLogger('lightning_logs') 
        trainer = pl.Trainer( 
            devices=self.gpus, accelerator="gpu", precision=self.precision,
            max_steps=self.pretrain_steps, logger=logger,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            accumulate_grad_batches=self.accumulate_grad_batches,
            callbacks=[TrainRequeue(
                nb_epochs_before_requeue=self.nb_epochs_before_requeue,
                checkpoint=self.p_checkpoint)])
        
        if os.path.exists(self.p_checkpoint):
            logging.info('Loading checkpoint {}...'.format(self.p_checkpoint))
            ckpt_path = self.p_checkpoint
            self = S3LSpeech.load_from_checkpoint(
                self.p_checkpoint, config=self.config)
            self.load_random_shift()
        else:
            ckpt_path = None
        
        logging.info('Pretraining model ...')
        self.steps_per_epoch = len(pretrain_dataloader)
        self.configure_optimizers = self.pretrain_optimizers
        self.training_step = self.pretrain_model
        trainer.fit(self, 
            train_dataloaders=pretrain_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=ckpt_path)
        trainer.save_checkpoint(self.p_checkpoint)
        
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
        self = S3LSpeech.load_from_checkpoint(
            self.checkpoint, config=self.config)
        trainer = pl.Trainer(
            devices=self.gpus, accelerator="gpu", precision=self.precision,) 
        trainer.test(
            self, dataloaders=test_dataloader, ckpt_path=self.checkpoint)

    def sample_negatives_flat(self, y, nums):
        """
        Negatives to calculate the contrastive loss
        usage: sampled_negatives, _ = self.sample_negatives(targets, targets.size(1))

        [1] https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/nemo/collections/asr/models/wav2vec/wav2vec_model.py
        """
        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        assert bsz == 1 and tsz == sum(nums)  # fake batch dim
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # cross_high = tsz * bsz

        neg_idxs_l = []
        idx_start = 0
        with torch.no_grad():
            for i, num_i in enumerate(nums):
                assert num_i > 1, f"{bsz, tsz, fsz}"

                assert self.n_negatives > 0
                tszs_i = buffered_arange(
                    num_i).unsqueeze(-1).expand(-1, self.n_negatives).flatten()

                high_i = num_i
                neg_idxs_i = torch.randint(
                    low=0, high=high_i - 1, size=(self.n_negatives * num_i,))
                neg_idxs_i[neg_idxs_i >= tszs_i] += 1

                neg_idxs_i += idx_start
                idx_start += num_i

                neg_idxs_l.append(neg_idxs_i)

                assert self.cross_sample_negatives == 0

        neg_idxs = torch.cat(neg_idxs_l)
        assert neg_idxs.ndim == 1

        negs = y[neg_idxs]
        negs = negs.view(
            bsz, sum(nums), 
            self.n_negatives + self.cross_sample_negatives, fsz).permute(
            2, 0, 1, 3)  # to NxBxTxC
        return negs, neg_idxs
    
    def ema_update(self, ema_module, new_module, m):
        with torch.no_grad():
            for param_q, param_k in zip(
                new_module.parameters(), ema_module.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def momentum_schedule(self, base_value, final_value, max_steps, step, type):
        if type == 'linear':
            if step <= max_steps:
                cur_value = base_value + (final_value - base_value) * (step / max_steps)
            else:
                cur_value = final_value
            return cur_value
        elif type == 'cosine':
            if step <= max_steps:
                cur_value = final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * step / max_steps))
            else:
                cur_value = final_value
            return cur_value
        else:
            raise ValueError('unknown scheduler type: {}'.format(type))
    
    def contrastive_loss(
        self,
        logits: torch.tensor,
        targets: torch.tensor,
        negatives: torch.tensor,
        ):
        """
        Args:
            logits: Model activations
            targets: The true target representations
            negatives: Sampled negatives from the input
        
        Returns:
            output loss values

        [1] https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/nemo/collections/asr/losses/wav2vecloss.py
        """

        # Calculate similarity between logits and all targets, returning FxBxT
        similarity_scores = self._calculate_similarity(
            logits, negatives, targets)

        # Create targets of size B*T
        similarity_targets = logits.new_zeros(
            similarity_scores.size(1) * similarity_scores.size(2), 
            dtype=torch.long)

        # Transpose similarity scores to (T*B)xF for loss
        similarity_scores = similarity_scores.transpose(0, 2)
        similarity_scores = similarity_scores.reshape(
            -1, similarity_scores.size(-1))

        contrastive_loss = F.cross_entropy(
            similarity_scores, similarity_targets, reduction='mean')
        
        accuracy = None
        with torch.no_grad():
            if similarity_scores.numel() == 0:
                corr = 0
                count = 0
                accuracy = float('nan')
            else:
                assert similarity_scores.dim() > 1, similarity_scores.shape
                max = similarity_scores.argmax(-1) == 0
                min = similarity_scores.argmin(-1) == 0
                both = max & min
                corr = max.long().sum().item() - both.long().sum().item()
                count = float(max.numel())
                accuracy = corr / count

        return contrastive_loss, accuracy

    def _calculate_similarity(self, logits, negatives, targets):
        neg_is_pos = (targets == negatives).all(-1)
        targets = targets.unsqueeze(0)
        targets = torch.cat([targets, negatives], dim=0)
        logits = torch.cosine_similarity(
            logits.float(), targets.float(), dim=-1).type_as(logits)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        return logits

class S3LSpeechDT(pl.LightningModule):
    """
    Super class for downstream tasks using EfficientSSL as pretraining class
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config.data_dir
        self.checkpoint = os.path.join(config.data_dir, config.checkpoint)
        self.p_checkpoint = os.path.join(config.data_dir, config.p_checkpoint)
        self.freeze_finetuning = config.freeze_finetuning
        self.pretraining = config.pretraining
        self.precision = config.precision
        self.gpus = config.gpus

        self.nb_epochs_before_requeue = config.nb_epochs_before_requeue
        self.pretrain_steps = config.pretrain_steps
        self.finetune_steps = config.finetune_steps
        self.pretrain_lr = config.pretrain_lr
        self.finetune_lr = config.finetune_lr
        self.check_val_every_n_epoch = config.check_val_every_n_epoch
        self.accumulate_grad_batches = config.accumulate_grad_batches 
        self.optimizer_betas = config.optimizer_betas
        self.optimizer_weight_decay = config.optimizer_weight_decay

        self.n_negatives = config.n_negatives 
        self.cross_sample_negatives = config.cross_sample_negatives
        self.logit_temp = config.logit_temp
        
        self.beam_size = config.beam_size
        self.lm_weight = config.lm_weight
        self.word_score = config.word_score

        self.former_step = -1
        self.val_step_losses = []

    def forward(self, x):
        x_student = x
        return x_student

    def validation_step(self, batch, batch_idx):
        self.shared_eval_step(batch, batch_idx, "Validation")

    def test_step(self, batch, batch_idx):
        self.shared_eval_step(batch, batch_idx, "Test")

    def on_validation_epoch_end(self):
        self.shared_eval_epoch_end("Validation")

    def on_test_epoch_end(self):
        self.shared_eval_epoch_end("Test")
    
    def finetune_optimizers(self):
        logging.info("Configuring finetune optimizers ...")
        adam = torch.optim.AdamW(
            self.parameters(), 
            lr=self.finetune_lr, 
            betas=self.optimizer_betas, 
            weight_decay=self.optimizer_weight_decay)
        scheduler = {
            'scheduler' : TriStageLRScheduler(
                optimizer=adam, init_lr_scale=1e-6, final_lr_scale=1e-6,
                warmup_updates=0.1 * self.finetune_steps,
                hold_updates=0.8 * self.finetune_steps,
                decay_updates=0.1 * self.finetune_steps,
                ),
            'interval': 'step',
            'frequency': 1
            }
        return {'optimizer' : adam, 'lr_scheduler' : scheduler}
                    
def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]

def efficient_pretrain_requeue():
    config = S3LSpeechConfig()
    librispeech_train_all = datasets.load_librispeech(
        data_root=config.data_dir, data_sets='all_train',
        max_batch_length=config.batch_length)
    librispeech_train_100 = datasets.load_librispeech(
        data_root=config.data_dir, data_sets='train_clean_100',
        max_batch_length=config.batch_length)
    librispeech_dev_other = datasets.load_librispeech(
        data_root=config.data_dir, data_sets='dev_other',
        max_batch_length=config.batch_length)
    
    config = S3LSpeechConfig()
    config.freeze_finetuning = False
    config.pretraining=True
    model = S3LSpeech(config)
    logging.info('\n\n--Experiment with efficient SSL config {0}'.format(
        config.__dict__))
    model.pretrain_model_requeue(
        pretrain_dataloader=librispeech_train_all,
        finetune_dataloader=librispeech_train_100,
        val_dataloader=librispeech_dev_other)