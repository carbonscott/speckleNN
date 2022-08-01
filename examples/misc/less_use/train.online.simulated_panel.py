#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.datasets.simulated_panels import OnlineDataset      , ConfigDataset
from deepprojection.model                     import OnlineSiameseModel , ConfigSiameseModel
from deepprojection.trainer                   import OnlineTrainer      , ConfigTrainer
from deepprojection.validator                 import OnlineLossValidator, ConfigValidator
from deepprojection.encoders.convnet          import Hirotaka0122       , ConfigEncoder
from deepprojection.utils                     import EpochManager       , MetaLog
from datetime import datetime
from simulated_panel_preprocess import DatasetPreprocess
import socket

# Set up parameters for an experiment...
size_sample    = 6000 * 3
size_batch     = 40
alpha          = 1.0
online_shuffle = True
lr             = 1e-3

# Clarify the purpose of this experiment...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Online training.

            Sample size    : {size_sample}
            Batch  size    : {size_batch}
            Alpha          : {alpha}
            Online shuffle : {online_shuffle}
            lr             : {lr}

            """

# [[[ LOGGING ]]]
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

# Create a metalog to the log file, explaining the purpose of this run...
metalog = MetaLog( comments = comments )
metalog.report()


# [[[ DATASET ]]]
# Config the dataset...
exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ConfigDataset.NOHIT, ConfigDataset.BACKGROUND ]
panels         = [ 0, 1 ,2, 3 ]
config_dataset = ConfigDataset( fl_csv         = 'simulated_datasets.csv',
                                size_sample    = size_sample, 
                                resize         = None,
                                seed           = 0,
                                panels         = panels,
                                isflat         = False,
                                istrain        = True,
                                frac_train     = 0.7,
                                trans          = None,
                                exclude_labels = exclude_labels, )

# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
dataset_preproc = DatasetPreprocess(config_dataset)
dataset_preproc.apply()
size_y, size_x = dataset_preproc.get_panelsize()

# Define training set...
config_dataset.report()
with OnlineDataset(config_dataset) as dataset_train:
    # Define validation set...
    config_dataset.istrain = False
    config_dataset.report()
    dataset_validate = OnlineDataset(config_dataset)


    # [[[ IMAGE ENCODER ]]]
    # Config the encoder...
    dim_emb = 128
    config_encoder = ConfigEncoder( dim_emb = dim_emb,
                                    size_y  = size_y,
                                    size_x  = size_x,
                                    isbias  = True )
    encoder = Hirotaka0122(config_encoder)


    # [[[ MODEL ]]]
    # Config the model...
    config_siamese = ConfigSiameseModel( alpha = alpha, encoder = encoder, )
    model = OnlineSiameseModel(config_siamese)

    # Initialize weights...
    def init_weights(module):
        if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
            module.weight.data.normal_(mean = 0.0, std = 0.02)
    model.apply(init_weights)


    # [[[ CHECKPOINT ]]]
    DRCCHKPT         = "chkpts"
    prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
    fl_chkpt         = f"{timestamp}.train.chkpt"
    path_chkpt       = os.path.join(prefixpath_chkpt, fl_chkpt)


    # [[[ TRAINER ]]]
    # Config the trainer...
    config_train = ConfigTrainer( path_chkpt     = path_chkpt,
                                  num_workers    = 0,
                                  batch_size     = size_batch,
                                  pin_memory     = True,
                                  shuffle        = False,
                                  online_shuffle = online_shuffle,
                                  is_logging     = False,
                                  method         = 'semi-hard', 
                                  ## method         = 'random-semi-hard', 
                                  ## method         = 'random', 
                                  lr             = lr, )

    # Training...
    trainer = OnlineTrainer(model, dataset_train, config_train)


    # [[[ VALIDATOR ]]]
    config_validator = ConfigValidator( path_chkpt     = None,
                                        num_workers    = 0,
                                        batch_size     = size_batch,
                                        pin_memory     = True,
                                        shuffle        = False,
                                        online_shuffle = online_shuffle,
                                        is_logging     = False,
                                        method         = 'semi-hard', 
                                        ## method         = 'random-semi-hard', 
                                        ## method         = 'random', 
                                        lr             = lr, 
                                        isflat         = False, )  # Conv2d input needs one more dim for batch

    validator = OnlineLossValidator(model, dataset_validate, config_validator)


    # [[[ EPOCH MANAGER ]]]
    max_epochs = 360
    epoch_manager = EpochManager(trainer = trainer, validator = validator, max_epochs = max_epochs)
    epoch_manager.run()
