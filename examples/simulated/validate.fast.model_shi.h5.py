#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
import socket
import pickle
import tqdm
import numpy as np

from deepprojection.datasets.lite    import SPIDataset, SPIOnlineDataset
from deepprojection.plugins          import DatasetModifierForModelShi
from deepprojection.model            import Shi2019Model
from deepprojection.encoders.convnet import Shi2019
from deepprojection.trainer          import SimpleTrainer, ConfigTrainer
from deepprojection.validator        import SimpleValidator, ConfigValidator
from deepprojection.utils            import MetaLog, init_logger, split_dataset, set_seed, NNSize, TorchModelAttributeParser, Config, EpochManager

from datetime import datetime

## from image_preprocess_faulty import DatasetPreprocess
## from image_preprocess_half import DatasetPreprocess
# from image_preprocess_one_four import DatasetPreprocess
from image_preprocess_faulty_pnccd import DatasetPreprocess

class MacroMetric:
    def __init__(self, res_dict):
        self.res_dict = res_dict


    def reduce_confusion(self, label):
        ''' Given a label, reduce multiclass confusion matrix to binary
            confusion matrix.
        '''
        res_dict    = self.res_dict
        labels      = res_dict.keys()
        labels_rest = [ i for i in labels if not i == label ]

        # Early return if non-exist label is passed in...
        if not label in labels: 
            print(f"label {label} doesn't exist!!!")
            return None

        # Obtain true positive...
        tp = len(res_dict[label][label])
        fp = sum( [ len(res_dict[label][i]) for i in labels_rest ] )
        tn = sum( sum( len(res_dict[i][j]) for j in labels_rest ) for i in labels_rest )
        fn = sum( [ len(res_dict[i][label]) for i in labels_rest ] )

        return tp, fp, tn, fn


    def get_metrics(self, label):
        # Early return if non-exist label is passed in...
        confusion = self.reduce_confusion(label)
        if confusion is None: return None

        # Calculate metrics...
        tp, fp, tn, fn = confusion
        accuracy    = (tp + tn) / (tp + tn + fp + fn)
        precision   = tp / (tp + fp)
        recall      = tp / (tp + fn)
        specificity = tn / (tn + fp) if tn + fp > 0 else None
        f1_inv      = (1 / precision + 1 / recall)
        f1          = 2 / f1_inv

        return accuracy, precision, recall, specificity, f1


# [[[ SEED ]]]
seed = 0
set_seed(seed)

# [[[ CONFIG ]]]
timestamp = "2022_1206_1629_17"    # 6Q5U
## timestamp = "2022_1207_1538_03"
frac_train = 0.5
frac_validate = 0.5

prob_threshold = 0.5

size_sample_test = 1000
size_sample_per_class = None
size_batch = 100
online_shuffle = True
trans = None


# Configure the location to run the job...## 
drc_cwd = os.getcwd()

init_logger(log_name = 'validate.query.test', timestamp = timestamp, returns_timestamp = False)


# Clarify the purpose of this experiment...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Online training.

            Sample size (test)     : {size_sample_test}
            Sample size (per class) : {size_sample_per_class}
            Batch  size             : {size_batch}
            Online shuffle          : {online_shuffle}

            """


# [[[ DATASET ]]]
# Set up parameters for an experiment...
drc_dataset   = 'fastdata.h5'
fl_dataset    = '1RWT.pnccd.pickle'    # Raw, just give it a try
path_dataset  = os.path.join(drc_dataset, fl_dataset)

# Load raw data...
with open(path_dataset, 'rb') as fh:
    dataset_list = pickle.load(fh)


# Split data...
data_train   , data_val_and_test = split_dataset(dataset_list     , frac_train   , seed = None)
data_validate, data_test         = split_dataset(data_val_and_test, frac_validate, seed = None)

# Define the test set
dataset_test = SPIOnlineDataset( dataset_list = data_test, 
                                 size_sample  = size_sample_test,
                                 size_sample_per_class = size_sample_per_class, 
                                 trans = trans, 
                                 seed  = None, )


# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
img_orig            = dataset_test[0][0][0]   # idx, fetch img
dataset_preproc     = DatasetPreprocess(img_orig)
trans               = dataset_preproc.config_trans()
dataset_test.trans  = trans
img_trans           = dataset_test[0][0][0]


dataset_test_modified = DatasetModifierForModelShi(dataset_test)


device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# [[[ IMAGE ENCODER ]]]
# Config the encoder...
size_y, size_x = img_trans.shape[-2:]
config_encoder = Config( name   = "Shi2019",
                         size_y = size_y,
                         size_x = size_x,
                         isbias = True )
encoder = Shi2019(config_encoder)


# [[[ MODEL ]]]
# Config the model...
config_model = Config( name = "Model", encoder = encoder, )
model = Shi2019Model(config_model)
model.init_params(from_timestamp = timestamp)

model.to(device)

# Validate an epoch...
# Load model state...
model.eval()
loader_test = torch.utils.data.DataLoader( dataset_test_modified, shuffle     = True, 
                                                                  pin_memory  = True, 
                                                                  batch_size  = size_batch,
                                                                  num_workers = 1 )

# New container to store validation result (thus res_dict) for each label...
labels = set([i[1] for i in dataset_test_modified])
res_dict = {}
for label in labels: res_dict[label] = { i : [] for i in labels }

# Train each batch...
batch = tqdm.tqdm(enumerate(loader_test), total = len(loader_test), disable = False)
for step_id, entry in batch:
    batch_imgs, batch_labels, batch_metadata = entry
    batch_imgs = batch_imgs.to(device, dtype = torch.float)
    batch_labels = batch_labels[:, None].to(device, dtype = torch.float)

    with torch.no_grad():
        batch_logit, loss = model.forward(batch_imgs, batch_labels)

    batch_label_pred = torch.where(torch.sigmoid(batch_logit) > prob_threshold, torch.ones_like(batch_logit), torch.zeros_like(batch_logit))

    for i in range(len(batch_label_pred)):
        label_pred = batch_label_pred[i].item()
        label_true = batch_labels[i].item()
        res_dict[label_pred][label_true].append( batch_metadata[i] )

# Get macro metrics...
macro_metric = MacroMetric(res_dict)

# Formating purpose...
disp_dict = { 0 : "not-single",
              1 : "single-hit",
            }

# Just print out the timestamp
print(timestamp)

# Report multiway classification...
msgs = []
for label_pred in sorted(labels):
    disp_text = disp_dict[label_pred]
    msg = f"{disp_text}  |"
    for label_real in sorted(labels):
        num = len(res_dict[label_pred][label_real])
        msg += f"{num:>12d}"

    metrics = macro_metric.get_metrics(label_pred)
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

## msg_headerbar = "-" * len(msgs[0])
## print(msg_headerbar)
for msg in msgs:
    print(msg)
