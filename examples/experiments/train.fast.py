#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
import socket
import pickle
import tqdm

from deepprojection.datasets.lite    import SPIDataset         , SPIOnlineDataset
from deepprojection.model            import OnlineSiameseModel , ConfigSiameseModel
from deepprojection.trainer          import OnlineTrainer      , ConfigTrainer
from deepprojection.validator        import OnlineLossValidator, ConfigValidator
from deepprojection.encoders.convnet import Hirotaka0122       , ConfigEncoder
from deepprojection.utils            import EpochManager       , MetaLog, init_logger, split_dataset
from datetime import datetime
from image_preprocess import DatasetPreprocess

# [[[ CONFIG ]]]
timestamp_prev = None
frac_train = 0.5
frac_validate = 0.5

lr = 1e-3
alpha = 2.0
seed = 0

size_sample_train     = 500
size_sample_validate  = 500
size_sample_per_class = 60
size_batch            = 200
online_shuffle        = True
trans                 = None

# [[[ LOGGING ]]]
timestamp = init_logger(log_name = 'train', returns_timestamp = True)

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

# Create a metalog to the log file, explaining the purpose of this run...
metalog = MetaLog( comments = comments )
metalog.report()


# [[[ DATASET ]]]
# Set up parameters for an experiment...
drc_dataset   = 'fastdata'
fl_dataset    = '0000.fastdata'    # Raw, just give it a try
path_dataset  = os.path.join(drc_dataset, fl_dataset)

# Load raw data...
with open(path_dataset, 'rb') as fh:
    dataset_list = pickle.load(fh)

# Split data...
data_train   , data_val_and_test = split_dataset(dataset_list     , frac_train   , seed = seed)
data_validate, data_test         = split_dataset(data_val_and_test, frac_validate, seed = seed)

# Define the training set
dataset_train = SPIOnlineDataset( dataset_list = data_train, 
                                  size_sample  = size_sample_train,
                                  size_sample_per_class = size_sample_per_class, 
                                  trans = trans, 
                                  seed  = seed, )
dataset_train.report()

# Define the training set
dataset_validate = SPIOnlineDataset( dataset_list = data_validate, 
                                     size_sample  = size_sample_train,
                                     size_sample_per_class = size_sample_per_class, 
                                     trans = trans, 
                                     seed  = seed, )
dataset_validate.report()


# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
img_orig               = dataset_train[0][0][0]   # idx, fetch img
dataset_preproc        = DatasetPreprocess(img_orig)
trans                  = dataset_preproc.config_trans()
dataset_train.trans    = trans
dataset_validate.trans = trans
img_trans              = dataset_train[0][0][0]


# [[[ IMAGE ENCODER ]]]
# Config the encoder...
dim_emb        = 128
size_y, size_x = img_trans.shape[-2:]
config_encoder = ConfigEncoder( dim_emb = dim_emb,
                                size_y  = size_y,
                                size_x  = size_x,
                                isbias  = True )
encoder = Hirotaka0122(config_encoder)


# [[[ MODEL ]]]
# Config the model...
config_siamese = ConfigSiameseModel( alpha = alpha, encoder = encoder, )
model = OnlineSiameseModel(config_siamese)
model.init_params(from_timestamp = timestamp_prev)


# [[[ CHECKPOINT ]]]
drc_cwd          = os.getcwd()
DRCCHKPT         = "chkpts"
prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
fl_chkpt         = f"{timestamp}.train.chkpt"
path_chkpt       = os.path.join(prefixpath_chkpt, fl_chkpt)


# [[[ TRAINER ]]]
# Config the trainer...
config_train = ConfigTrainer( path_chkpt     = path_chkpt,
                              num_workers    = 1,
                              batch_size     = size_batch,
                              pin_memory     = True,
                              shuffle        = False,
                              is_logging     = False,
                              online_shuffle = online_shuffle,
                              method         = 'random-semi-hard', 
                              lr             = lr, 
                              tqdm_disable   = True)
trainer = OnlineTrainer(model, dataset_train, config_train)


# [[[ VALIDATOR ]]]
config_validator = ConfigValidator( path_chkpt     = None,
                                    num_workers    = 1,
                                    batch_size     = size_batch,
                                    pin_memory     = True,
                                    shuffle        = False,
                                    is_logging     = False,
                                    online_shuffle = online_shuffle,
                                    method         = 'random-semi-hard', 
                                    lr             = lr,
                                    tqdm_disable   = True)  # Conv2d input needs one more dim for batch
validator = OnlineLossValidator(model, dataset_validate, config_validator)


# [[[ TRAIN EPOCHS ]]]

loss_train_hist = []
loss_validate_hist = []
loss_min_hist = []

# [[[ EPOCH MANAGER ]]]
epoch_manager = EpochManager( trainer   = trainer,
                              validator = validator,
                              timestamp = timestamp, )
max_epochs = 400
freq_save = 5
for epoch in tqdm.tqdm(range(max_epochs), disable=False):
    loss_train, loss_validate, loss_min = epoch_manager.run_one_epoch(epoch = epoch, returns_loss = True)

    loss_train_hist.append(loss_train)
    loss_validate_hist.append(loss_validate)
    loss_min_hist.append(loss_min)

    # if epoch % freq_save == 0: 
    #     epoch_manager.save_model_parameters()
    #     epoch_manager.save_model_gradients()
    #     epoch_manager.save_state_dict()
