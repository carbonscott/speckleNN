#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.datasets.images  import OnlineDataset      , ConfigDataset
from deepprojection.model            import OnlineSiameseModel , ConfigSiameseModel
from deepprojection.trainer          import OnlineTrainer      , ConfigTrainer
from deepprojection.validator        import OnlineLossValidator, ConfigValidator
from deepprojection.encoders.convnet import Hirotaka0122       , ConfigEncoder
from deepprojection.utils            import EpochManager       , MetaLog
from datetime import datetime
from image_preprocess import DatasetPreprocess
## from image_no_reg_preprocess import DatasetPreprocess
import socket

# Set up parameters for an experiment...
## fl_csv                = 'datasets.simple.csv'
## ## size_sample_train     = 100 * 1
## ## size_sample_validate  = 225 * 1
## size_sample_train     = 2000
## size_sample_validate  = 2000
## exclude_labels        = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ConfigDataset.BACKGROUND ]
## frac_train            = 0.25
## dataset_usage         = 'train'

fl_csv               = 'datasets.binary.csv'
## size_sample_train    =  60 * 1
## size_sample_validate = 225 * 1
size_sample_train    = 2000
size_sample_validate = 2000
size_sample_per_class = 40
exclude_labels       = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ConfigDataset.MULTI, ConfigDataset.BACKGROUND ]
frac_train           = 0.25
frac_validate        = None
dataset_usage        = 'train'

size_batch     = 20
alpha          = 2.0
online_shuffle = True
lr             = 1e-3
seed           = 0

# Clarify the purpose of this experiment...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Online training.

            Sample size (train)     : {size_sample_train}
            Sample size (validate)  : {size_sample_validate}
            Sample size (per class) : {size_sample_per_class}
            Batch  size             : {size_batch}
            Alpha                   : {alpha}
            Online shuffle          : {online_shuffle}
            lr                      : {lr}

            """

# [[[ LOGGING ]]]
# Create a timestamp to name the log file...
now = datetime.now()
timestamp = now.strftime("%Y_%m%d_%H%M_%S")

# Configure the location to run the job...
drc_cwd = os.getcwd()

# Config logging behaviors
logging.basicConfig( format="%(asctime)s %(levelname)s %(name)-35s - %(message)s",
                     datefmt="%m/%d/%Y %H:%M:%S",
                     level=logging.INFO, )
logger = logging.getLogger(__name__)

# Create a metalog to the log file, explaining the purpose of this run...
metalog = MetaLog( comments = comments )
metalog.report()

# [[[ DATASET ]]]
# Config the dataset...
config_dataset = ConfigDataset( fl_csv                = fl_csv,
                                size_sample           = size_sample_train, 
                                mode                  = 'image',
                                mask                  = None,
                                resize                = None,
                                seed                  = seed,
                                isflat                = False,
                                dataset_usage         = dataset_usage,
                                trans                 = None,
                                frac_train            = frac_train,
                                frac_validate         = frac_validate,
                                size_sample_per_class = size_sample_per_class,
                                exclude_labels        = exclude_labels, )

# Define the training set
dataset_train = OnlineDataset(config_dataset)

# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
img_orig, _         = dataset_train.get_img_and_label(0)
dataset_preproc     = DatasetPreprocess(img_orig)
trans               = dataset_preproc.config_trans()
dataset_train.trans = trans
img_trans, _        = dataset_train.get_img_and_label(0)

dataset_train.report()

# Report training set...
config_dataset.trans = trans
config_dataset.report()

# Define validation set...
config_dataset.size_sample           = size_sample_validate
config_dataset.dataset_usage         = 'validate'
config_dataset.size_sample_per_class = None
config_dataset.report()
dataset_validate = OnlineDataset(config_dataset)



# Define test set...
config_dataset.size_sample           = size_sample_validate
config_dataset.dataset_usage         = 'test'
config_dataset.size_sample_per_class = None
config_dataset.report()
dataset_validate = OnlineDataset(config_dataset)
