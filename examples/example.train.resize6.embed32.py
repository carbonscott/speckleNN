#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.experiments import SPIImgDataset, SiameseDataset, ConfigDataset
from deepprojection.model       import SiameseModel , ConfigSiameseModel
from deepprojection.trainer     import Trainer      , ConfigTrainer

from datatime import datetime

# Create a timestamp to name the log file...
now = datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")

# Config logging behaviors
logging.basicConfig( filename = f"{timestamp}.log",
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
                                debug          = True, 
                                resize         = (resize_y, resize_x),
                                exclude_labels = [ ConfigDataset.UNKNOWN ],
                                debug          = True )
dataset_train = SiameseDataset(config_dataset)

# Get image size...
spiimg = SPIImgDataset(config_dataset)
size_y, size_x = spiimg.get_imagesize(0)

# Try different margin (alpha) for Siamese net...
for i, alpha in enumerate((1.0, )):
    # Config the model...
    config_siamese = ConfigSiameseModel(alpha = alpha, dim_emb = 32, size_y = size_y, size_x = size_x)
    model          = SiameseModel(config_siamese)

    # Initialize model...
    model.apply(init_weights)

    # Config the trainer...
    drc_cwd = os.getcwd()
    path_chkpt = os.path.join(drc_cwd, f"trained_model.{i:02d}.chkpt")
    config_train = TrainerConfig( path_chkpt  = path_chkpt,
                                  num_workers = 1,
                                  batch_size  = 100,
                                  max_epochs  = 30,
                                  lr          = 0.001, 
                                  debug       = debug, )

    # Training...
    trainer = Trainer(model, dataset_train, config_train)
    trainer.train()
