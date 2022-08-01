#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.datasets.simulated_pnccd_detector import MultiwayQueryset      , ConfigDataset
from deepprojection.model                             import SiameseModelCompare   , ConfigSiameseModel
from deepprojection.validator                         import MultiwayQueryValidator, ConfigValidator
from deepprojection.encoders.convnet                  import Hirotaka0122          , ConfigEncoder
from deepprojection.datasets                          import transform
from deepprojection.utils                             import MetaLog
from deepprojection.plugins                           import PsanaImg
from simulated_pnccd_panel_preprocess                 import DatasetPreprocess
import itertools
import socket

# Create a timestamp to name the log file...
timestamp = "2022_0722_2156_30"

# Set up parameters for an experiment...
fl_csv         = "simulated.pnccd_panel.v2.datasets.csv"
num_query      = 2000
frac_train     = 0.70
frac_validate  = None
dataset_usage  = 'test'

size_batch     = 200
online_shuffle = True
lr             = 1e-3
seed           = 0


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

# Load PsanaImg...
exp           = 'amo06516'
run           = '102'
mode          = 'idx'
detector_name = 'Camp.0:pnCCD.0'

psana_img = PsanaImg( exp           = exp,
                      run           = run,
                      mode          = mode,
                      detector_name = detector_name, )

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

# Create a metalog to the log file, explaining the purpose of this run...
metalog = MetaLog( comments = comments )
metalog.report()


# Config the dataset...
exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ConfigDataset.NOHIT, ConfigDataset.BACKGROUND ]
config_dataset = ConfigDataset( fl_csv         = fl_csv,
                                size_sample    = num_query,
                                seed           = 0,
                                isflat         = False,
                                frac_train     = frac_train,
                                frac_validate  = None,
                                dataset_usage  = dataset_usage,
                                exclude_labels = exclude_labels,
                                psana_img      = psana_img, )

# Define the training set
dataset_validate = MultiwayQueryset(config_dataset)

# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
img_orig               = dataset_validate[0][0][0]    # idx, fetch img, fetch from batch
dataset_preproc        = DatasetPreprocess(img_orig)
trans                  = dataset_preproc.config_trans()
dataset_validate.trans = trans
img_trans              = dataset_validate[0][0][0]    # idx, fetch img, fetch from batch

idx_list = list(itertools.chain(*dataset_validate.queryset))
dataset_validate.cache_img(idx_list)

# Define training set...
config_dataset.trans = trans
config_dataset.report()

# Fetch checkpoint directory...
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
