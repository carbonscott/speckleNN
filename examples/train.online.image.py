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
import socket

# Set up parameters for an experiment...
fl_csv         = 'datasets.simple.csv'
size_sample    = 200
size_batch     = 20
alpha          = 2.0
online_shuffle = True
lr             = 1e-3
frac_train     = 0.7
seed           = 0

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
## exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP ]
exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ConfigDataset.BACKGROUND ]
## exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ConfigDataset.NOHIT, ConfigDataset.BACKGROUND ]
config_dataset = ConfigDataset( fl_csv         = fl_csv,
                                size_sample    = size_sample, 
                                mode           = 'image',
                                mask           = None,
                                resize         = None,
                                seed           = seed,
                                isflat         = False,
                                istrain        = True,
                                trans          = None,
                                frac_train     = frac_train,
                                exclude_labels = exclude_labels, )

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
config_dataset.istrain = False
config_dataset.report()
dataset_validate = OnlineDataset(config_dataset)


# [[[ IMAGE ENCODER ]]]
# Config the encoder...
dim_emb        = 128
size_y, size_x = img_trans.shape
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
                              is_logging     = False,
                              online_shuffle = online_shuffle,
                              method         = 'semi-hard', 
                              lr             = lr, )

# Training...
trainer = OnlineTrainer(model, dataset_train, config_train)


# [[[ VALIDATOR ]]]
config_validator = ConfigValidator( path_chkpt     = None,
                                    num_workers    = 0,
                                    batch_size     = size_batch,
                                    pin_memory     = True,
                                    shuffle        = False,
                                    is_logging     = False,
                                    online_shuffle = online_shuffle,
                                    method         = 'semi-hard', 
                                    lr             = lr,
                                    isflat         = False, )  # Conv2d input needs one more dim for batch
validator = OnlineLossValidator(model, dataset_validate, config_validator)


# [[[ EPOCH MANAGER ]]]
max_epochs = 360
epoch_manager = EpochManager(trainer = trainer, validator = validator, max_epochs = max_epochs)
epoch_manager.run()
