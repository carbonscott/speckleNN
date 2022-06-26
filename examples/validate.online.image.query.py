#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.datasets.images  import MultiwayQueryset      , ConfigDataset
from deepprojection.model            import SiameseModelCompare   , ConfigSiameseModel
from deepprojection.validator        import MultiwayQueryValidator, ConfigValidator
from deepprojection.encoders.convnet import Hirotaka0122          , ConfigEncoder
from deepprojection.utils            import MetaLog
## from image_preprocess                import DatasetPreprocess
from image_no_reg_preprocess import DatasetPreprocess
import socket

# Create a timestamp to name the log file...
timestamp = "2022_0625_1704_31"

# Set up parameters for an experiment...
fl_csv         = 'datasets.simple.csv'
num_query      = 1000
frac_train     = 0.25
frac_validate  = None
size_batch     = 40
online_shuffle = True
lr             = 1e-3
dataset_usage  = 'test'
seed           = 0

exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ConfigDataset.BACKGROUND ]
## exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ConfigDataset.MULTI, ConfigDataset.BACKGROUND ]

# Comment this verification...
hostname = socket.gethostname()
comments = f""" 
            Hostname: {hostname}.

            Online training.

            Number (query) : {num_query}
            Batch  size    : {size_batch}
            Online shuffle : {online_shuffle}
            lr             : {lr}

            """

# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
fl_log         = f"{timestamp}.validate.query.test.log"
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
config_dataset = ConfigDataset( fl_csv         = fl_csv,
                                size_sample    = num_query,
                                mode           = 'image',
                                resize         = None,
                                seed           = seed,
                                isflat         = False,
                                dataset_usage  = dataset_usage,
                                trans          = None,
                                frac_train     = frac_train,
                                exclude_labels = exclude_labels, )

# Define the training set
dataset_validate = MultiwayQueryset(config_dataset)

# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
img_orig, _         = dataset_validate.get_img_and_label(0)
dataset_preproc     = DatasetPreprocess(img_orig)
trans               = dataset_preproc.config_trans()
dataset_validate.trans = trans
img_trans, _        = dataset_validate.get_img_and_label(0)

# Define validation set...
config_dataset.trans = trans
config_dataset.report()

DRCCHKPT = "chkpts"
prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)

# Config the encoder...
dim_emb = 128
size_y, size_x = img_trans.shape
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
                                    batch_size  = size_batch,
                                    pin_memory  = True,
                                    shuffle     = False,
                                    isflat      = False,
                                    lr          = lr, )

validator = MultiwayQueryValidator(model, dataset_validate, config_validator)
validator.validate()
