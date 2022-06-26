#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.datasets.simulated_square_detector import OnlineDataset      , ConfigDataset
from deepprojection.model                              import OnlineSiameseModel , ConfigSiameseModel
from deepprojection.trainer                            import OnlineTrainer      , ConfigTrainer
from deepprojection.validator                          import OnlineLossValidator, ConfigValidator
from deepprojection.encoders.convnet                   import Hirotaka0122       , ConfigEncoder
from deepprojection.utils                              import EpochManager       , MetaLog
from simulated_square_detector_preprocess              import DatasetPreprocess
from datetime import datetime
import socket

# Set up parameters for an experiment...
fl_csv               = "simulated.square_detector.datasets.pdb_sampled.10.csv"
## fl_csv               = "simulated.square_detector.datasets.6Q5U.csv"
size_sample_train    = 2000
size_sample_validate = 2000
size_batch           = 25
frac_train           = 0.5
frac_validate        = None
dataset_usage        = 'train'
alpha                = 2.0
online_shuffle       = True
lr                   = 1e-3
seed                 = 0

# Clarify the purpose of this experiment...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Online training.

            Sample size (train)    : {size_sample_train}
            Sample size (validate) : {size_sample_train}
            Batch  size            : {size_batch}
            Alpha                  : {alpha}
            Online shuffle         : {online_shuffle}
            lr                     : {lr}

            """

# [[[ LOGGING ]]]
# Create a timestamp to name the log file...
now = datetime.now()
timestamp = now.strftime("%Y_%m%d_%H%M_%S")

# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
fl_log         = f"{timestamp}.train.log"
DRCLOG         = "logs"
prefixpath_log = os.path.join(drc_cwd, DRCLOG)
if not os.path.exists(prefixpath_log): os.makedirs(prefixpath_log)
path_log = os.path.join(prefixpath_log, fl_log)

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
exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ConfigDataset.NOHIT, ConfigDataset.BACKGROUND ]
config_dataset = ConfigDataset( fl_csv         = fl_csv,
                                size_sample    = size_sample_train, 
                                seed           = seed,
                                isflat         = False,
                                dataset_usage  = dataset_usage,
                                frac_train     = frac_train,
                                frac_validate  = frac_validate,
                                trans          = None,
                                exclude_labels = exclude_labels, )

# Define training set...
config_dataset.report()
dataset_train = OnlineDataset(config_dataset)

# Define validation set...
config_dataset.size_sample   = size_sample_validate
config_dataset.dataset_usage = 'validate'
config_dataset.report()
dataset_validate = OnlineDataset(config_dataset)
