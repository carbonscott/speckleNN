#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.datasets.images  import MultiwayQueryset      , ConfigDataset
from deepprojection.model            import SiameseModelCompare   , ConfigSiameseModel
from deepprojection.validator        import MultiwayQueryValidator, ConfigValidator
from deepprojection.encoders.convnet import Hirotaka0122          , ConfigEncoder
from deepprojection.datasets         import transform
from image_preprocess                import DatasetPreprocess

# Create a timestamp to name the log file...
timestamp = "20220323222231"

# Validate mode...
istrain = False
mode_validate = 'train' if istrain else 'test'

# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
fl_log         = f"{timestamp}.validate.query.{mode_validate}.log"
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
exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP ]
config_dataset = ConfigDataset( fl_csv         = 'datasets.csv',
                                size_sample    = 2000, 
                                mode           = 'image',
                                resize         = None,
                                seed           = 0,
                                isflat         = False,
                                istrain        = istrain,
                                trans          = None,
                                frac_train     = 0.8,
                                exclude_labels = exclude_labels, )

# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
dataset_preproc = DatasetPreprocess(config_dataset)
dataset_preproc.apply()
size_y, size_x = dataset_preproc.get_imgsize()

# Define validation set...
config_dataset.report()
dataset_validate = MultiwayQueryset(config_dataset)

DRCCHKPT = "chkpts"
prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)

# Config the encoder...
dim_emb = 128
config_encoder = ConfigEncoder( dim_emb = dim_emb,
                                size_y  = size_y,
                                size_x  = size_x,
                                isbias  = True )
encoder = Hirotaka0122(config_encoder)

# Set up the model
config_siamese = ConfigSiameseModel( encoder = encoder, )
model = SiameseModelCompare(config_siamese)

# Read chkpt from a trainig
fl_chkpt = f"{timestamp}.train.chkpt"
path_chkpt = os.path.join(prefixpath_chkpt, fl_chkpt)
config_validator = ConfigValidator( path_chkpt  = path_chkpt,
                                    num_workers = 1,
                                    batch_size  = 40,
                                    pin_memory  = True,
                                    shuffle     = False,
                                    isflat      = False,
                                    lr          = 1e-3, )

validator = MultiwayQueryValidator(model, dataset_validate, config_validator)
validator.validate()
