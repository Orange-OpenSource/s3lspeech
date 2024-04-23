# Software Name : s3lspeech
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.

import argparse
import logging
import datasets
from models.ssl import efficient_pretrain_requeue
from models.asr import asr_finetune_requeue

from config.s3lspeech import (
    DATA_DIR,
    LOG_FILE,
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train S3LSpeech model')
    parser.add_argument("--log_file", default=LOG_FILE, type=str)
    parser.add_argument("--run", default='pretrain', type=str)
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file, format='%(asctime)s %(levelname)s %(message)s', 
        filemode='w', level=logging.INFO)
    if args.run == "download":
        datasets.get_librispeech(data_root=DATA_DIR)
    elif args.run == "pretrain":
        efficient_pretrain_requeue()
    elif args.run == "finetune":
        asr_finetune_requeue()
