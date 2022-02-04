#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.datasets.experiments import SPIImgDataset, SiameseDataset, ConfigDataset
from deepprojection.model                import SiameseModel , ConfigSiameseModel
from deepprojection.validator            import Validator      , ConfigValidator
from deepprojection.encoders.linear      import SimpleEncoder, ConfigEncoder
from utils import read_log


# Create a timestamp to name the log file...
## timestamp = "20220203115233"
timestamp = "20220203150247"

# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
fl_log         = f"{timestamp}.validate.log"
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

# Config the dataset...
resize_y, resize_x = 6, 6
config_dataset = ConfigDataset( fl_csv         = 'datasets.csv',
                                size_sample    = 1000, 
                                resize         = (resize_y, resize_x),
                                exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ], )
                                ## exclude_labels = [ ConfigDataset.NOHIT, ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ], )
dataset_validate = SiameseDataset(config_dataset)


# Get image size
spiimg = SPIImgDataset(config_dataset)
size_y, size_x = spiimg.get_imagesize(0)

DRCCHKPT = "chkpts"
prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)

# Config the encoder...
dim_emb = 32
config_encoder = ConfigEncoder( dim_emb = dim_emb,
                                size_y  = size_y,
                                size_x  = size_x,
                                isbias  = True )
encoder = SimpleEncoder(config_encoder)

# Set up the model
alpha   = 1.0
config_siamese = ConfigSiameseModel( alpha   = alpha, encoder = encoder, )
model = SiameseModel(config_siamese)

# Read chkpt from a trainig
fl_chkpt = f"{timestamp}.train.chkpt"
path_chkpt = os.path.join(prefixpath_chkpt, fl_chkpt)
config_validator = ConfigValidator( path_chkpt  = path_chkpt,
                                    num_workers = 1,
                                    batch_size  = 200,
                                    max_epochs  = 1,    # Epoch = 1 for validate
                                    lr          = 0.001, )


validator = Validator(model, dataset_validate, config_validator)
validator.validate()
