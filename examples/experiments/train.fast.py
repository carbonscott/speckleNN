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
from deepprojection.utils            import EpochManager       , MetaLog, init_logger, split_dataset, set_seed

from datetime import datetime
## from image_preprocess import DatasetPreprocess
from image_preprocess_faulty import DatasetPreprocess
## from image_preprocess_rotation import DatasetPreprocess

# [[[ SEED ]]]
seed = 0
set_seed(seed)


# [[[ CONFIG ]]]
timestamp_prev = None
frac_train     = 0.5
frac_validate  = 0.5

logs_triplets = True

lr = 1e-3

alpha = 0.02
## alpha = 0.03336201
## alpha = 0.05565119
## alpha = 0.09283178
## alpha = 0.15485274
## alpha = 0.25830993
## alpha = 0.43088694
## alpha = 0.71876273
## alpha = 1.1989685
## alpha = 2.0

## size_sample_per_class_train    = 10
## size_sample_per_class_train    = 20
## size_sample_per_class_train    = 40
size_sample_per_class_train    = 60
size_sample_train              = size_sample_per_class_train * 100
size_sample_validate           = size_sample_train // 2
size_sample_per_class_validate = size_sample_per_class_train // 2
size_batch                     = 20
trans                          = None

# [[[ LOGGING ]]]
timestamp = init_logger(log_name = 'train', returns_timestamp = True)
print(timestamp)

# Clarify the purpose of this experiment...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Online training.

            Sample size (train)               : {size_sample_train}
            Sample size (validate)            : {size_sample_validate}
            Sample size (per class, train)    : {size_sample_per_class_train}
            Sample size (per class, validate) : {size_sample_per_class_validate}
            Batch  size                       : {size_batch}
            Alpha                             : {alpha}
            lr                                : {lr}

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
data_train   , data_val_and_test = split_dataset(dataset_list     , frac_train   , seed = None)
data_validate, data_test         = split_dataset(data_val_and_test, frac_validate, seed = None)

# Define the training set
dataset_train = SPIOnlineDataset( dataset_list          = data_train, 
                                  size_sample           = size_sample_train,
                                  size_sample_per_class = size_sample_per_class_train, 
                                  trans                 = trans, 
                                  seed                  = None, )
dataset_train.report()

# Define the training set
dataset_validate = SPIOnlineDataset( dataset_list          = data_validate, 
                                     size_sample           = size_sample_train,
                                     size_sample_per_class = size_sample_per_class_validate, 
                                     trans                 = trans, 
                                     seed                  = None, )
dataset_validate.report()


# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
img_orig               = dataset_train[0][0][0]   # idx, fetch img
dataset_preproc        = DatasetPreprocess(img_orig)
trans                  = dataset_preproc.config_trans()
dataset_train.trans    = trans
dataset_validate.trans = trans
img_trans              = dataset_train[0][0][0]

dataset_train.cache_dataset()
dataset_validate.cache_dataset()

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
config_siamese = ConfigSiameseModel( alpha   = alpha, 
                                     encoder = encoder, 
                                     seed    = None     # No need to reset seed.  It has been set in data split.
                                   )
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
config_train = ConfigTrainer( path_chkpt        = path_chkpt,
                              num_workers       = 1,
                              batch_size        = size_batch,
                              pin_memory        = True,
                              shuffle           = False,
                              logs_triplets     = logs_triplets,
                              method            = 'random-semi-hard', 
                              lr                = lr, 
                              tqdm_disable      = True)
trainer = OnlineTrainer(model, dataset_train, config_train)


# [[[ VALIDATOR ]]]
config_validator = ConfigValidator( path_chkpt        = None,
                                    num_workers       = 1,
                                    batch_size        = size_batch,
                                    pin_memory        = True,
                                    shuffle           = False,
                                    logs_triplets     = logs_triplets,
                                    method            = 'random-semi-hard', 
                                    lr                = lr,
                                    tqdm_disable      = True)  # Conv2d input needs one more dim for batch
validator = OnlineLossValidator(model, dataset_validate, config_validator)


# [[[ TRAIN EPOCHS ]]]

loss_train_hist    = []
loss_validate_hist = []
loss_min_hist      = []

# [[[ EPOCH MANAGER ]]]
epoch_manager = EpochManager( trainer   = trainer,
                              validator = validator,
                              timestamp = timestamp, )
max_epochs = 1000
freq_save = 5
for epoch in tqdm.tqdm(range(max_epochs), disable=False):
    if epoch > 0:
        epoch_manager.trainer.config.logs_triplets   = False
        epoch_manager.validator.config.logs_triplets = False

    loss_train, loss_validate, loss_min = epoch_manager.run_one_epoch(epoch = epoch, returns_loss = True)

    loss_train_hist.append(loss_train)
    loss_validate_hist.append(loss_validate)
    loss_min_hist.append(loss_min)

    # if epoch % freq_save == 0: 
    #     epoch_manager.save_model_parameters()
    #     epoch_manager.save_model_gradients()
    #     epoch_manager.save_state_dict()
