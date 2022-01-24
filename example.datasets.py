#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Check if dataset class Siamese produces the expected outcomes.

'''

import torch
import torch.nn as nn
from deepprojection.dataset import SPIImgDataset, SiameseDataset
from deepprojection.model   import SiameseModel, SiameseConfig
from deepprojection.trainer import TrainerConfig, Trainer
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataLoader


class VizMetadata():

    def __init__(self, size_y, size_x, figsize, **kwargs):
        self.size_y  = size_y
        self.size_x  = size_x
        self.figsize = figsize

        for k, v in kwargs.items(): setattr(self, k, v)


class VizSiameseOutput:
    def __init__(self, img_anchor, img_pos, img_neg, emb_anchor, emb_pos, emb_neg, viz_metadata): 
        self.viz_metadata = viz_metadata
        size_y, size_x = viz_metadata.size_y, viz_metadata.size_x
        self.figsize = viz_metadata.figsize

        self.img_anchor = img_anchor.reshape(size_y, size_x)
        self.img_pos    = img_pos.reshape(size_y, size_x)
        self.img_neg    = img_neg.reshape(size_y, size_x)
        self.emb_anchor = emb_anchor.detach().numpy()
        self.emb_pos    = emb_pos.detach().numpy()
        self.emb_neg    = emb_neg.detach().numpy()

        self.title_anchor = viz_metadata.title_anchor if getattr(viz_metadata, "title_anchor") else "anchor"
        self.title_pos    = viz_metadata.title_pos    if getattr(viz_metadata, "title_pos")    else "pos"
        self.title_neg    = viz_metadata.title_neg    if getattr(viz_metadata, "title_neg")    else "neg"

        self.fig,           \
        self.ax_img_anchor, \
        self.ax_img_pos,    \
        self.ax_img_neg,    \
        self.ax_emb_anchor, \
        self.ax_emb_pos,    \
        self.ax_emb_neg,    \
        = self.create_panels()

        return None


    def create_panels(self):
        fig, ((ax_img_anchor, ax_img_pos, ax_img_neg), \
              (ax_emb_anchor, ax_emb_pos, ax_emb_neg)) \
              = plt.subplots(nrows = 2, ncols = 3, sharey = False, figsize = self.figsize)
        return fig, ax_img_anchor, ax_img_pos, ax_img_neg, \
                    ax_emb_anchor, ax_emb_pos, ax_emb_neg


    def plot_img_anchor(self, title = ""): 
        self.ax_img_anchor.imshow(self.img_anchor, vmax = 100, vmin = 0)
        self.ax_img_anchor.set_title(title)


    def plot_img_pos(self, title = ""): 
        self.ax_img_pos.imshow(self.img_pos, vmax = 100, vmin = 0)
        self.ax_img_pos.set_title(title)


    def plot_img_neg(self, title = ""): 
        self.ax_img_neg.imshow(self.img_neg, vmax = 100, vmin = 0)
        self.ax_img_neg.set_title(title)


    def plot_emb_anchor(self, title = ""): 
        self.ax_emb_anchor.plot(self.emb_anchor)
        self.ax_emb_anchor.set_title(title)


    def plot_emb_pos(self, title = ""): 
        self.ax_emb_pos.plot(self.emb_pos)
        self.ax_emb_pos.set_title(title)


    def plot_emb_neg(self, title = ""): 
        self.ax_emb_neg.plot(self.emb_neg)
        self.ax_emb_neg.set_title(title)


    def show(self): 
        self.plot_img_anchor(self.title_anchor)
        self.plot_img_pos(self.title_pos)
        self.plot_img_neg(self.title_neg)
        self.plot_emb_anchor(self.title_anchor)
        self.plot_emb_pos(self.title_pos)
        self.plot_emb_neg(self.title_neg)
        plt.show()




def init_weights(module):
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean = 0.5, std = 0.02)

fl_csv = 'datasets.csv'
size_sample = 1000
debug = True
dataset_train = SiameseDataset(fl_csv, size_sample, debug = debug)

# Get image size
spiimg = SPIImgDataset(fl_csv)
size_y, size_x = spiimg.get_imagesize(0)

config_train = TrainerConfig( checkpoint_path = None,
                              num_workers     = 1,    # Visualization purpose
                              batch_size      = 1,
                              max_epochs      = 4,
                              lr              = 0.0001, )

# Load siamese model
config_siamese = SiameseConfig(alpha = 0.5, size_y = size_y, size_x = size_x)
model = SiameseModel(config_siamese)
model.apply(init_weights)

# Train each epoch
for epoch in tqdm.tqdm(range(config_train.max_epochs)):
    loader_train = DataLoader( dataset_train, shuffle     = True, 
                                              pin_memory  = True, 
                                              batch_size  = config_train.batch_size,
                                              num_workers = config_train.num_workers )
    losses = []

    # Train each batch
    batch = tqdm.tqdm(enumerate(loader_train), total = len(loader_train))
    for step_id, entry in batch:
        if debug: img_anchor, img_pos, img_neg, label_anchor, \
                  title_anchor, title_pos, title_neg = entry
        else:     img_anchor, img_pos, img_neg, label_anchor = entry

        ## print(title_anchor)

        for i in range(len(label_anchor)):
            emb_anchor, emb_pos, emb_neg, _ = model.forward(img_anchor[i], img_pos[i], img_neg[i])
            viz_metadata = VizMetadata(size_y = size_y, size_x = size_x, figsize = (12*3,6*3),
                                       title_anchor = title_anchor[i],
                                       title_pos    = title_pos[i],
                                       title_neg    = title_neg[i],)
            viz = VizSiameseOutput(img_anchor[i], img_pos[i], img_neg[i], emb_anchor, emb_pos, emb_neg, viz_metadata)
            viz.show()
