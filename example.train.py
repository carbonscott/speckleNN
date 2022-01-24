#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from deepprojection.dataset import SPIImgDataset, SiameseDataset
from deepprojection.model   import Siamese
from deepprojection.trainer import TrainerConfig, Trainer
import matplotlib.pyplot as plt


class ImageCompare:
    def __init__(self, img1, img2): 
        self.img1 = img1
        self.img2 = img2

        self.fig, self.ax_img1, self.ax_img2 = self.create_panels()

        return None

    def create_panels(self):
        fig, (ax_img1, ax_img2) = plt.subplots(ncols = 2, sharey = False)
        return fig, ax_img1, ax_img2

    def plot_img1(self, title = ""): 
        self.ax_img1.imshow(self.img1, vmax = 100, vmin = 0)
        self.ax_img1.set_title(title)

    def plot_img2(self, title = ""): 
        self.ax_img2.imshow(self.img2, vmax = 1.0, vmin = 0.8)
        self.ax_img2.set_title(title)

    def show(self): plt.show()


def init_weights(module):
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean = 0.0, std = 0.02)

fl_csv = 'datasets.csv'
size_sample = 100
seed = 4
dataset_train = SiameseDataset(fl_csv, size_sample, seed)

# Get image size
spiimg = SPIImgDataset(fl_csv)
size_y, size_x = spiimg.get_imagesize(0)
size_img = size_y * size_x

alpha = 0.5
model = Siamese(alpha, size_img)
model.apply(init_weights)

config_train = TrainerConfig( checkpoint_path = None,
                              num_workers     = 0,
                              batch_size      = 64,
                              max_epochs      = 2,
                              alpha           = 0.1,
                              lr              = 0.001, )

trainer = Trainer(model, dataset_train, None, config_train)
trainer.train()
