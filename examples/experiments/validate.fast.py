#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
import socket
import pickle
import tqdm
import numpy as np

from deepprojection.datasets.lite    import MultiwayQueryset
from deepprojection.model            import SiameseModelCompare, ConfigSiameseModel
from deepprojection.validator        import MultiwayQueryValidator, ConfigValidator
from deepprojection.encoders.convnet import Hirotaka0122, ConfigEncoder
from deepprojection.utils            import EpochManager, MetaLog, init_logger, split_dataset, ConfusionMatrix, set_seed

from datetime         import datetime
from image_preprocess import DatasetPreprocess
## from image_preprocess_faulty import DatasetPreprocess

# [[[ SEED ]]]
seed = 0
set_seed(seed)


# [[[ CONFIG ]]]
timestamp     = "2022_1121_1935_37"
frac_train    = 0.5
frac_validate = 0.5

lr   = 1e-3

# Define the test set
size_sample_test      = 1000
size_sample_per_class = None
size_batch            = 100
online_shuffle        = True
trans                 = None

# Initialize a log file...
init_logger(log_name = 'validate.query.test', timestamp = timestamp, returns_timestamp = False)


# [[[ DATASET ]]]
# Set up parameters for an experiment...
drc_dataset   = 'fastdata'
fl_dataset    = '0000.fastdata'
path_dataset  = os.path.join(drc_dataset, fl_dataset)

# Load raw data...
with open(path_dataset, 'rb') as fh:
    dataset_list = pickle.load(fh)

# Split data...
data_train   , data_val_and_test = split_dataset(dataset_list     , frac_train   , seed = None)
data_validate, data_test         = split_dataset(data_val_and_test, frac_validate, seed = None)

dataset_test = MultiwayQueryset( dataset_list          = data_test, 
                                 size_sample           = size_sample_test,
                                 size_sample_per_class = size_sample_per_class, 
                                 trans                 = trans, 
                                 seed                  = None, )

# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
img_orig            = dataset_test[0][0][0]   # idx, fetch img
dataset_preproc     = DatasetPreprocess(img_orig)
trans               = dataset_preproc.config_trans()
dataset_test.trans  = trans
img_trans           = dataset_test[0][0][0]


# [[[ MODEL ]]]
# Config the encoder...
dim_emb        = 128
size_y, size_x = img_trans.shape[-2:]
config_encoder = ConfigEncoder( dim_emb = dim_emb,
                                size_y  = size_y,
                                size_x  = size_x,
                                isbias  = True )
encoder = Hirotaka0122(config_encoder)


# Set up the model
config_siamese_test = ConfigSiameseModel( encoder = encoder, )
model_test = SiameseModelCompare(config_siamese_test)
model_test.init_params(from_timestamp = timestamp)



# [[[ VALIDATE ]]]
config_tester = ConfigValidator( num_workers = 1,
                                 batch_size  = size_batch,
                                 pin_memory  = True,
                                 shuffle     = False,
                                 isflat      = False,
                                 lr          = lr, )
tester = MultiwayQueryValidator(model_test, dataset_test, config_tester)
batch_metadata_query_list, batch_metadata_support_list, batch_dist_support_list = tester.validate(returns_details = True)


# [[[ CONFUSION MATRIX ]]]

idx_min_value_list = np.argmin(batch_dist_support_list, axis = 1)
batch_metadata_support_selected_list = []
for idx_batch in range(len(batch_metadata_support_list)):
    metadata_support_selected_list = []
    for idx_example in range(len(batch_metadata_support_list[0][0])):
        idx_min_value = idx_min_value_list[idx_batch][idx_example]
        metadata_support_selected = batch_metadata_support_list[idx_batch][idx_min_value][idx_example]
        metadata_support_selected_list.append(metadata_support_selected)
    batch_metadata_support_selected_list.append(metadata_support_selected_list)

labels = set([ metadata.split()[-1] for batch_metadata in batch_metadata_query_list for metadata in batch_metadata[0] ])

# New container to store validation result (thus res_dict) for each label...
res_dict = {}
for label in labels: res_dict[label] = { i : [] for i in labels }

for idx_batch in range(len(batch_metadata_support_list)):
    for idx_example in range(len(batch_metadata_query_list[idx_batch][0])):
        metadata_true = batch_metadata_query_list[idx_batch][0][idx_example]
        metadata_pred = batch_metadata_support_selected_list[idx_batch][idx_example]
        label_true = metadata_true.split()[-1]
        label_pred = metadata_pred.split()[-1]
        res_dict[label_pred][label_true].append( (metadata_true, metadata_pred) )


# Get confusion matrix...
confusion_matrix = ConfusionMatrix(res_dict)

# Formating purpose...
disp_dict = { "0" : "not-sample",
              "1" : "single-hit",
              "2" : " multi-hit",
              "9" : "background",
            }

# Report multiway classification...
msgs = []
for label_pred in sorted(labels):
    disp_text = disp_dict[label_pred]
    msg = f"{disp_text}  |"
    for label_real in sorted(labels):
        num = len(res_dict[label_pred][label_real])
        msg += f"{num:>12d}"

    metrics = confusion_matrix.get_metrics(label_pred)
    for metric in metrics:
        msg += f"{metric:>12.2f}"
    msgs.append(msg)

msg_header = " " * (msgs[0].find("|") + 1)
for label in sorted(labels): 
    disp_text = disp_dict[label]
    msg_header += f"{disp_text:>12s}"

for header in [ "accuracy", "precision", "recall", "specificity", "f1" ]:
    msg_header += f"{header:>12s}"
print(msg_header)

msg_headerbar = "-" * len(msgs[0])
print(msg_headerbar)
for msg in msgs:
    print(msg)
