#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.datasets.experiments import SPIImgDataset, SiameseDataset, ConfigDataset
from deepprojection.model                import SiameseModel , ConfigSiameseModel
from deepprojection.trainer              import Trainer      , ConfigTrainer
from deepprojection.encoders.linear      import SimpleEncoder, ConfigEncoder

from datetime import datetime

# Global variable
DEBUG = True

# Create a timestamp to name the log file...
now = datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")

# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
fl_log         = f"{timestamp}.train.log"
DRCLOG         = "logs"
prefixpath_log = os.path.join(drc_cwd, DRCLOG)
if not os.path.exists(prefixpath_log): os.makedirs(prefixpath_log)
path_log = os.path.join(prefixpath_log, fl_log)

# Config logging behaviors
logging.basicConfig( filename = path_log,
                     filemode = 'w',
                     format="%(asctime)s %(levelname)s %(name)-35s - %(message)s",
                     datefmt="%m/%d/%Y %H:%M:%S",
                     level=logging.INFO, )
logger = logging.getLogger(__name__)

# Initialize weights...
def init_weights(module):
    if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
        module.weight.data.normal_(mean = 0.0, std = 0.02)

# Config the dataset...
resize_y, resize_x = 6, 6
resize = (resize_y, resize_x) if not None in (resize_y, resize_x) else ()
exclude_labels = [ ConfigDataset.NOHIT, ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP ]
## exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP ]
## exclude_labels = [ ConfigDataset.UNKNOWN ]
config_dataset = ConfigDataset( fl_csv         = 'datasets.csv',
                                size_sample    = 1000, 
                                resize         = resize,
                                exclude_labels = exclude_labels,
                                debug          = DEBUG )
dataset_train = SiameseDataset(config_dataset)

# Get image size...
spiimg = SPIImgDataset(config_dataset)
size_y, size_x = spiimg.get_imagesize(0)

# Try different margin (alpha) for Siamese net...
DRCCHKPT = "chkpts"
prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
if not os.path.exists(prefixpath_chkpt): os.makedirs(prefixpath_chkpt)

# Config the encoder...
dim_emb = 32
config_encoder = ConfigEncoder( dim_emb = dim_emb,
                                size_y  = size_y,
                                size_x  = size_x,
                                isbias  = True )
encoder = SimpleEncoder(config_encoder)

# Config the model...
alpha   = 1.0
config_siamese = ConfigSiameseModel( alpha   = alpha, 
                                     dim_emb = dim_emb, 
                                     size_y  = size_y, 
                                     size_x  = size_x,
                                     encoder = encoder, )
model = SiameseModel(config_siamese)

# Initialize model...
model.apply(init_weights)

# Config the trainer...
fl_chkpt = f"{timestamp}.train.chkpt"
path_chkpt = os.path.join(prefixpath_chkpt, fl_chkpt)
config_train = ConfigTrainer( path_chkpt  = path_chkpt,
                              num_workers = 1,
                              batch_size  = 200,
                              max_epochs  = 10,
                              lr          = 0.001, 
                              debug       = DEBUG, )

# Training...
trainer = Trainer(model, dataset_train, config_train)
trainer.train()
