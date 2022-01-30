#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.datasets import SPIImgDataset, SiameseDataset, ConfigDataset
from deepprojection.model    import SiameseModel , ConfigSiameseModel
from deepprojection.trainer  import Trainer      , ConfigTrainer

from datetime import datetime

# Global variable
DEBUG = True

# Create a timestamp to name the log file...
now = datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")

# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
fl_log         = f"train.{timestamp}.log"
prefixpath_log = os.path.join(drc_cwd, "results.train")
if not os.path.exists(prefixpath_log): os.makedirs(prefixpath_log)
path_log = os.path.join(prefixpath_log, fl_log)

# Config logging behaviors
logging.basicConfig( filename = path_log,
                     filemode = 'w',
                     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                     datefmt="%m/%d/%Y %H:%M:%S",
                     level=logging.INFO, )
logger = logging.getLogger(__name__)

# Initialize weights...
def init_weights(module):
    if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
        module.weight.data.normal_(mean = 0.0, std = 0.02)

# Config the dataset...
resize_y, resize_x = 6, 6
config_dataset = ConfigDataset( fl_csv         = 'datasets.csv',
                                size_sample    = 1000, 
                                resize         = (resize_y, resize_x),
                                exclude_labels = [ ConfigDataset.UNKNOWN ],
                                debug          = DEBUG )
dataset_train = SiameseDataset(config_dataset)

# Get image size...
spiimg = SPIImgDataset(config_dataset)
size_y, size_x = spiimg.get_imagesize(0)

# Try different margin (alpha) for Siamese net...
prefixpath_chkpt = os.path.join(drc_cwd, "chkpts.train")
if not os.path.exists(prefixpath_chkpt): os.makedirs(prefixpath_chkpt)
for i, alpha in enumerate((1.0, )):
    # Config the model...
    config_siamese = ConfigSiameseModel(alpha = alpha, dim_emb = 32, size_y = size_y, size_x = size_x)
    model          = SiameseModel(config_siamese)

    # Initialize model...
    model.apply(init_weights)

    # Config the trainer...
    fl_chkpt = f"train.{timestamp}.{i:02d}.chkpt"
    path_chkpt = os.path.join(prefixpath_chkpt, fl_chkpt)
    config_train = ConfigTrainer( path_chkpt  = path_chkpt,
                                  num_workers = 1,
                                  batch_size  = 100,
                                  max_epochs  = 15,
                                  lr          = 0.001, 
                                  debug       = DEBUG, )

    # Training...
    trainer = Trainer(model, dataset_train, config_train)
    trainer.train()
