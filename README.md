<!---
# Software Name : s3lspeech
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
--->

### Installation

A docker file based on NVidia [23.04](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-04.html#rel-23-04) image is available under _docker/Dockerfile_, with all the dependencies and libraries for running experiments. Otherwise, follow a manual install:

Create a virtual environment and activate it
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-tk
sudo pip3 install -U virtualenv
virtualenv --system-site-packages -p python3 ~/torch21
source ~/torch21/bin/activate
```

Install Python packages
```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### Experiments

For replicating results, settings are stored in _config/s3lspeech.py_. 

Download datasets 
```bash
python3 main.py --run download
```
Pretrain the model
```bash
python3 main.py --run pretrain
```
Finetune the model for ASR
```bash
python3 main.py --run finetune
```

Results are stored in the log _s3lspeech_results.pt_log_. The pretrained and finetuned checkpoints are stored under _data/_.

### License
This project is licensed under the terms of the [MIT](https://opensource.org/licenses/MIT) license. See the LICENSE file for more information. 