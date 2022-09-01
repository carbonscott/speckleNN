#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.datasets.simulated_square_detector import MultiwayQueryset      , ConfigDataset
from deepprojection.model                              import SiameseModelCompare   , ConfigSiameseModel
from deepprojection.validator                          import MultiwayQueryValidator, ConfigValidator
from deepprojection.encoders.convnet                   import Hirotaka0122          , ConfigEncoder
from deepprojection.datasets                           import transform
from deepprojection.utils                              import MetaLog
from simulated_square_detector_preprocess              import DatasetPreprocess
import socket

# Create a timestamp to name the log file...
timestamp = "2022_0511_1247_15"

# Set up parameters for an experiment...
fl_csv         = "simulated.square_detector.datasets.pdb_not_sampled.10.csv"
size_sample    = 2000
size_batch     = 20
lr             = 1e-3
frac_train     = 0.0
seed           = 0

# Comment this verification...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Sample size    : {size_sample}
            Batch  size    : {size_batch}
            lr             : {lr}

            Apply model to completed datasets, e.g. not in training or testing.
            """

# Validate mode...
istrain = False
mode_validate = 'train' if istrain else 'test'

# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
fl_log         = f"{timestamp}.validate.query.{mode_validate}.testsets.log"
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
exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ConfigDataset.NOHIT, ConfigDataset.BACKGROUND ]
config_dataset = ConfigDataset( fl_csv            = fl_csv,
                                size_sample       = size_sample, 
                                resize            = None,
                                seed              = seed,
                                isflat            = False,
                                istrain           = istrain,
                                frac_train        = frac_train,
                                exclude_labels    = exclude_labels, )

# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
dataset_preproc = DatasetPreprocess(config_dataset)
dataset_preproc.apply()
size_y, size_x = dataset_preproc.get_imgsize()

# Define validation set...
config_dataset.report()
dataset_validate = MultiwayQueryset(config_dataset)

# Fetch checkpoint directory...
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
                                    batch_size  = size_batch,
                                    pin_memory  = True,
                                    shuffle     = False,
                                    isflat      = False,
                                    lr          = lr, )

validator = MultiwayQueryValidator(model, dataset_validate, config_validator)
validator.validate()
