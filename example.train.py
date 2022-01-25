#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from deepprojection.dataset import SPIImgDataset, SiameseDataset
from deepprojection.model   import SiameseModel, SiameseConfig
from deepprojection.trainer import TrainerConfig, Trainer
import os


def init_weights(module):
    if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
        print(module)
        module.weight.data.normal_(mean = 0.0, std = 0.02)

fl_csv = 'datasets.csv'
size_sample = 20
debug = True
dataset_train = SiameseDataset(fl_csv, size_sample, debug = debug)

# Get image size
spiimg = SPIImgDataset(fl_csv)
size_y, size_x = spiimg.get_imagesize(0)

config_siamese = SiameseConfig(alpha = 1.0, size_y = size_y, size_x = size_x)
model = SiameseModel(config_siamese)
model.apply(init_weights)

drc_cwd = os.getcwd()
path_chkpt = os.path.join(drc_cwd, "trained_model.chkpt")
config_train = TrainerConfig( path_chkpt  = path_chkpt,
                              num_workers = 1,
                              batch_size  = 20,
                              max_epochs  = 1,
                              lr          = 0.001, 
                              debug       = debug, )

trainer = Trainer(model, dataset_train, config_train)
trainer.train()
