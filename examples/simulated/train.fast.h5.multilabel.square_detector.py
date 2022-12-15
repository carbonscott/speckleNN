#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
import socket
import pickle
import tqdm

from deepprojection.datasets.lite    import SPIDataset                , TripletCandidate
from deepprojection.model            import OnlineTripletSiameseModel , ConfigSiameseModel
from deepprojection.trainer          import OnlineTripletTrainer      , ConfigTrainer
from deepprojection.validator        import OnlineTripletLossValidator, ConfigValidator
from deepprojection.encoders.convnet import Hirotaka0122              , ConfigEncoder
from deepprojection.utils            import EpochManager              , MetaLog, init_logger, split_dataset, set_seed

from datetime import datetime

from image_preprocess_faulty_sq import DatasetPreprocess


# [[[ SEED ]]]
seed = 0
set_seed(seed)


# [[[ CONFIG ]]]
timestamp_prev = None
## timestamp_prev = "2022_1129_2150_15"

frac_train    = 0.5
frac_validate = 0.5

lr = 1e-3
alpha = 0.05565119

num_sample_per_label_train    = 20
num_sample_train              = 60
num_sample_per_label_validate = 20
num_sample_validate           = 30

size_batch                 = 20
trans                      = None

# [[[ LOGGING ]]]
timestamp = init_logger(log_name = 'train', returns_timestamp = True, saves_log = True)
print(timestamp)

# Clarify the purpose of this experiment...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Online training.

            Sample size (train)                          : {num_sample_train}
            Sample size (validate)                       : {num_sample_validate}
            Sample size (candidates per class, train)    : {num_sample_per_label_train}
            Sample size (candidates per class, validate) : {num_sample_per_label_validate}
            Batch  size                                  : {size_batch}
            Alpha                                        : {alpha}
            lr                                           : {lr}
            seed                                         : {seed}

            """

# Create a metalog to the log file, explaining the purpose of this run...
metalog = MetaLog( comments = comments )
metalog.report()


# [[[ DATASET ]]]
# Set up parameters for an experiment...
drc_dataset   = 'fastdata.h5'
fl_dataset    = 'mini.sq.train.pickle'    # Raw, just give it a try
path_dataset  = os.path.join(drc_dataset, fl_dataset)

# Load raw data...
with open(path_dataset, 'rb') as fh:
    dataset_list = pickle.load(fh)

# Split data...
data_train   , data_val_and_test = split_dataset(dataset_list     , frac_train   , seed = None)
data_validate, data_test         = split_dataset(data_val_and_test, frac_validate, seed = None)

# Define the training set
dataset_train = TripletCandidate( dataset_list          = data_train, 
                                  num_sample            = num_sample_train,
                                  num_sample_per_label  = num_sample_per_label_train, 
                                  trans                 = None, )
# dataset_train.report()

# Define the training set
dataset_validate = TripletCandidate( dataset_list          = data_validate, 
                                     num_sample            = num_sample_validate,
                                     num_sample_per_label  = num_sample_per_label_validate, 
                                     trans                 = None, )
# dataset_validate.report()

# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
img_orig               = dataset_train[0][1][0][0]   # idx, fetch img
dataset_preproc        = DatasetPreprocess(img_orig)
trans                  = dataset_preproc.config_trans()
dataset_train.trans    = trans
dataset_validate.trans = trans
img_trans              = dataset_train[0][1][0][0]

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
config_siamese = ConfigSiameseModel( alpha = alpha, encoder = encoder, )
model = OnlineTripletSiameseModel(config_siamese)
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
                              num_workers       = 0,
                              batch_size        = size_batch,
                              pin_memory        = True,
                              shuffle           = False,
                              lr                = lr, 
                              logs_triplets     = True,
                              tqdm_disable      = True)
trainer = OnlineTripletTrainer(model, dataset_train, config_train)


# [[[ VALIDATOR ]]]
config_validator = ConfigValidator( path_chkpt        = None,
                                    num_workers       = 0,
                                    batch_size        = size_batch,
                                    pin_memory        = True,
                                    shuffle           = False,
                                    lr                = lr,
                                    logs_triplets     = True,
                                    tqdm_disable      = True)  # Conv2d input needs one more dim for batch
validator = OnlineTripletLossValidator(model, dataset_validate, config_validator)


loss_train_hist = []
loss_validate_hist = []
loss_min_hist = []

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
