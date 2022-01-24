#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from deepprojection.dataset import SPIImgDataset, SiameseDataset
from deepprojection.model   import SiameseModel, SiameseConfig
from deepprojection.trainer import TrainerConfig, Trainer
import matplotlib.pyplot as plt


def init_weights(module):
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean = 0.0, std = 0.02)

fl_csv = 'datasets.csv'
size_sample = 1000
debug = True
dataset_train = SiameseDataset(fl_csv, size_sample, debug = debug)

# Get image size
spiimg = SPIImgDataset(fl_csv)
size_y, size_x = spiimg.get_imagesize(0)

config_siamese = SiameseConfig(alpha = 0.5, size_y = size_y, size_x = size_x)
model = SiameseModel(config_siamese)
model.apply(init_weights)

config_train = TrainerConfig( checkpoint_path = None,
                              num_workers     = 1,
                              batch_size      = 100,
                              max_epochs      = 4,
                              lr              = 0.001, 
                              debug           = debug, )

trainer = Trainer(model, dataset_train, None, config_train)
trainer.train()
