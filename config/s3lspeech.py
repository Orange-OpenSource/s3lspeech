# Software Name : s3lspeech
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.

DATA_DIR = "data"
LOG_FILE = "s3lspeech_results.pt_log"
CHECKPT  = "s3lspeech_finetuned_0.ckpt"
PCHECKPT = "s3lspeech_pretrained_0.ckpt"

class S3LSpeechConfig:
    """
    Default configuration values for the Eficcient SSL model
    """
    def __init__(self):
        self.data_dir = DATA_DIR
        self.checkpoint = CHECKPT
        self.p_checkpoint = PCHECKPT
        self.freeze_finetuning = False
        self.pretraining = True
        self.precision = "bf16-mixed"
        self.gpus = 1

        self.nb_epochs_before_requeue = 30
        self.pretrain_steps = 12500
        self.finetune_steps = 60000
        self.pretrain_lr = 3e-4
        self.finetune_lr = 3e-4
        self.accumulate_grad_batches = 4
        self.check_val_every_n_epoch = 120 * self.accumulate_grad_batches
        self.batch_length = 18 * 60
        self.optimizer_betas = (0.9, 0.98)
        self.optimizer_weight_decay = 0.01

        self.n_negatives = 100
        self.cross_sample_negatives = 0
        self.logit_temp = 0.5

        #https://pytorch.org/audio/stable/tutorials/asr_inference_with_ctc_decoder_tutorial.html
        self.beam_size = 500
        self.lm_weight = 3.23
        self.word_score = -0.4

        self.n_mels = 128
        self.log_mel = True
        self.sample_rate = 16000
        self.win_length = 20 # ms
        self.hop_length = 10 # ms
        self.n_fft = int(self.sample_rate * self.win_length / 1000)
        self.noise_ratio = 0.5