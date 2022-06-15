#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from deepprojection.datasets.mosaic  import MultiwayQueryset      , ConfigDataset
from deepprojection.model            import SiameseModelCompare   , ConfigSiameseModel
from deepprojection.validator        import MultiwayQueryValidator, ConfigValidator
from deepprojection.encoders.convnet import Hirotaka0122          , ConfigEncoder
from deepprojection.utils            import MetaLog
from mosaic_preprocess               import DatasetPreprocess
import socket

# Create a timestamp to name the log file...
timestamp = "2022_0614_2306_10"

# Set up parameters for an experiment...
fl_csv         = 'datasets.simple.csv'
size_sample    = 1000
size_batch     = 40
online_shuffle = True
lr             = 1e-3
frac_train     = 0.0
seed           = 0
panels_ordered = [0, 1]

# Comment this verification...
hostname = socket.gethostname()
comments = f""" 
            Hostname: {hostname}.

            Online training.

            Sample size    : {size_sample}
            Batch  size    : {size_batch}
            Online shuffle : {online_shuffle}
            lr             : {lr}

            """

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

# Create a metalog to the log file, explaining the purpose of this run...
metalog = MetaLog( comments = comments )
metalog.report()

# Config the dataset...
exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP ]
## exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELPi, ConfigDataset.BACKGROUND ]
config_dataset = ConfigDataset( fl_csv         = fl_csv,
                                size_sample    = size_sample, 
                                psana_mode     = 'calib',
                                seed           = 0,
                                isflat         = False,
                                istrain        = istrain,
                                trans          = None,
                                panels_ordered = panels_ordered,
                                frac_train     = frac_train,
                                exclude_labels = exclude_labels, )

# Define the training set
dataset_validate = MultiwayQueryset(config_dataset)

# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
dataset_validate.MOSAIC_ON = False
img_orig, _                = dataset_validate.get_img_and_label(0)
panel_orig                 = img_orig[0]
dataset_validate.MOSAIC_ON = True
dataset_preproc            = DatasetPreprocess(panel_orig)
trans                      = dataset_preproc.config_trans()
dataset_validate.trans     = trans
mosaic_trans, _            = dataset_validate.get_img_and_label(0)

# Report training set...
config_dataset.trans = trans
config_dataset.report()

# Config the encoder...
dim_emb = 128
size_y, size_x = mosaic_trans.shape
config_encoder = ConfigEncoder( dim_emb = dim_emb,
                                size_y  = size_y,
                                size_x  = size_x,
                                isbias  = True )
encoder = Hirotaka0122(config_encoder)

# Set up the model
config_siamese = ConfigSiameseModel( encoder = encoder, )
model = SiameseModelCompare(config_siamese)

# Read chkpt from a trainig
DRCCHKPT         = "chkpts"
prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
fl_chkpt         = f"{timestamp}.train.chkpt"
path_chkpt       = os.path.join(prefixpath_chkpt, fl_chkpt)
config_validator = ConfigValidator( path_chkpt  = path_chkpt,
                                    num_workers = 1,
                                    batch_size  = size_batch,
                                    pin_memory  = True,
                                    shuffle     = False,
                                    isflat      = False,
                                    lr          = lr, )

validator = MultiwayQueryValidator(model, dataset_validate, config_validator)
validator.validate()
