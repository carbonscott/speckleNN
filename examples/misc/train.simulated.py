#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.datasets.simulated import SiameseDataset, ConfigDataset
from deepprojection.model              import SiameseModel , ConfigSiameseModel
from deepprojection.trainer            import Trainer      , ConfigTrainer
from datetime import datetime

# Global variable
DEBUG = True

# Create a timestamp to name the log file...
now = datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")

# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
fl_log         = f"{timestamp}.train.simulated.log"
DRCLOG         = "logs"
prefixpath_log = os.path.join(drc_cwd, DRCLOG)
if not os.path.exists(prefixpath_log): os.makedirs(prefixpath_log)
path_log = os.path.join(prefixpath_log, fl_log)

# Config logging behaviors
logging.basicConfig( filename = path_log,
                     filemode = 'w',
                     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                     datefmt="%m/%d/%Y %H:%M:%S",
                     level=logging.INFO, )
logger = logging.getLogger(__name__)

def init_weights(module):
    if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
        ## print(module)
        module.weight.data.normal_(mean = 0.0, std = 0.02)


drc_cwd      = os.getcwd()
fl_x_train   = "x_train.npy"
fl_y_train   = "y_train.npy"
path_x_train = os.path.join(drc_cwd, fl_x_train)
path_y_train = os.path.join(drc_cwd, fl_y_train)

exclude_labels = [ ConfigDataset.UNKNOWN ]
config_dataset = ConfigDataset( path_x_train = path_x_train,
                                path_y_train = path_y_train,
                                size_sample  = 1000,
                                debug        = DEBUG )
dataset_train = SiameseDataset(config_dataset)

# This metadata is hand-coded for quick verification
size_y, size_x = 128, 128

# Try different margin (alpha) for Siamese net
alpha = 1.0
dim_emb = 32
config_siamese = ConfigSiameseModel(alpha = alpha, dim_emb = dim_emb, size_y = size_y, size_x = size_x)
model = SiameseModel(config_siamese)
model.apply(init_weights)

# Config the trainer...
fl_chkpt = f"{timestamp}.train.simulated.chkpt"
DRCCHKPT = "chkpts"
prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
path_chkpt = os.path.join(prefixpath_chkpt, fl_chkpt)
config_train = ConfigTrainer( path_chkpt  = path_chkpt,
                              num_workers = 1,
                              batch_size  = 100,
                              max_epochs  = 15,
                              lr          = 0.001, 
                              debug       = DEBUG, )

trainer = Trainer(model, dataset_train, config_train)
trainer.train()
