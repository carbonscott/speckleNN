#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.datasets.simulated_square_detector import SimpleSet          , ConfigDataset
from deepprojection.model                              import EmbeddingModel     , ConfigSiameseModel
from deepprojection.validator                          import EmbeddingCalculator, ConfigValidator
from deepprojection.encoders.convnet                   import Hirotaka0122       , ConfigEncoder
from deepprojection.datasets                           import transform
from deepprojection.utils                              import MetaLog
from simulated_square_detector_preprocess              import DatasetPreprocess
import socket

# Create a timestamp to name the log file...
timestamp = "2022_0901_1140_43"

# Set up parameters for an experiment...
fl_csv            = "simulated_datasets_3iyf.csv"
size_sample_train = 4000
frac_train        = 0.01

dim_emb = 128
alpha   = 2
sigma   = 0.15 * 1    # ...Define Gaussian noise level

size_sample_per_class = None
frac_validate         = None
size_batch            = 400
dataset_usage         = 'test'
online_shuffle        = True
lr                    = 1e-3
seed                  = 0

# Clarify the purpose of this experiment...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Online training.

            Sample size (train)     : {size_sample_train}
            Sample size (per class) : {size_sample_per_class}
            Batch  size             : {size_batch}
            Alpha                   : {alpha}
            Online shuffle          : {online_shuffle}
            lr                      : {lr}

            """


# ___/ EMBEDDING CALCULATION \___
# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
basename       = f"{timestamp}.comp_emb"
fl_log         = f"{basename}.log"
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
config_dataset = ConfigDataset( fl_csv                = fl_csv,
                                size_sample           = size_sample_train, 
                                seed                  = seed,
                                frac_train            = frac_train,
                                frac_validate         = frac_validate,
                                size_sample_per_class = size_sample_per_class,
                                dataset_usage         = dataset_usage,
                                trans                 = None,
                                exclude_labels        = exclude_labels, )


# Define validation set...
dataset = SimpleSet(config_dataset)

# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
img_orig            = dataset[0][0][0]
dataset_preproc     = DatasetPreprocess(img_orig, sigma = sigma)
trans               = dataset_preproc.config_trans()
dataset.trans = trans
img_trans           = dataset[0][0][0]

# Report training set...
config_dataset.trans = trans
config_dataset.report()

DRCCHKPT = "chkpts"
prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)

# Config the encoder...
dim_emb        = 128
size_y, size_x = img_trans.shape
config_encoder = ConfigEncoder( dim_emb = dim_emb,
                                size_y  = size_y,
                                size_x  = size_x,
                                isbias  = True )
encoder = Hirotaka0122(config_encoder)

# Set up the model
config_siamese = ConfigSiameseModel( encoder = encoder, )
model = EmbeddingModel(config_siamese)

# Read chkpt from a trainig
fl_chkpt = f"{timestamp}.train.chkpt"
path_chkpt = os.path.join(prefixpath_chkpt, fl_chkpt)
config_validator = ConfigValidator( ## path_chkpt  = None,    # Use it for raw input (random weights)
                                    path_chkpt  = path_chkpt,
                                    num_workers = 1,
                                    batch_size  = size_batch,
                                    pin_memory  = True,
                                    shuffle     = False,
                                    isflat      = False,
                                    lr          = lr, )
validator = EmbeddingCalculator(model, dataset, config_validator)
embs = validator.run()

# Fetch embedding directory...
fl_emb = f"{basename}.pt"
DRCEMB = "embeds"
prefixpath_emb = os.path.join(drc_cwd, DRCEMB)
if not os.path.exists(prefixpath_emb): os.makedirs(prefixpath_emb)
path_emb = os.path.join(prefixpath_emb, fl_emb)

torch.save(embs, path_emb)
