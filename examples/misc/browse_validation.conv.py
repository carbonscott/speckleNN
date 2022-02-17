#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Check if dataset class Siamese produces the expected outcomes.
'''

import logging
logging.basicConfig( format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                     datefmt="%m/%d/%Y %H:%M:%S",
                     level=logging.INFO, )

import torch
from deepprojection.datasets.experiments import SPIImgDataset, SiameseDataset, ConfigDataset
from deepprojection.model                import SiameseModel, ConfigSiameseModel
from deepprojection.validator            import ConfigValidator, Validator
from deepprojection.encoders.convnet     import Hirotaka0122, ConfigEncoder
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataLoader
import os


class VizMetadata():

    def __init__(self, size_y, size_x, figsize, **kwargs):
        self.size_y  = size_y
        self.size_x  = size_x
        self.figsize = figsize

        for k, v in kwargs.items(): setattr(self, k, v)


class VizSiameseOutput:
    def __init__(self, img_anchor, img_pos, img_neg, emb_anchor, emb_pos, emb_neg, viz_metadata): 
        self.viz_metadata = viz_metadata
        size_y, size_x    = viz_metadata.size_y, viz_metadata.size_x
        self.figsize      = viz_metadata.figsize

        self.img_anchor = img_anchor.reshape(size_y, size_x)
        self.img_pos    = img_pos.reshape(size_y, size_x)
        self.img_neg    = img_neg.reshape(size_y, size_x)
        self.emb_anchor = emb_anchor.cpu().detach().numpy()
        self.emb_pos    = emb_pos.cpu().detach().numpy()
        self.emb_neg    = emb_neg.cpu().detach().numpy()

        self.title_anchor = viz_metadata.title_anchor              if hasattr(viz_metadata, "title_anchor") else "anchor"
        self.title_pos    = viz_metadata.title_pos                 if hasattr(viz_metadata, "title_pos")    else "pos"
        self.title_neg    = viz_metadata.title_neg                 if hasattr(viz_metadata, "title_neg")    else "neg"
        self.title_fig    = f"loss = {viz_metadata.title_fig:.4f}" if hasattr(viz_metadata, "title_fig")    else "init"

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
        plt.rcParams.update({'font.size': 18})
        plt.rcParams.update({'font.family' : 'sans-serif'})
        fig, ((ax_img_anchor, ax_img_pos, ax_img_neg), \
              (ax_emb_anchor, ax_emb_pos, ax_emb_neg)) \
              = plt.subplots(nrows = 2, ncols = 3, sharey = False, figsize = self.figsize)
        return fig, ax_img_anchor, ax_img_pos, ax_img_neg, \
                    ax_emb_anchor, ax_emb_pos, ax_emb_neg


    def plot_img_anchor(self, title = ""): 
        self.ax_img_anchor.imshow(self.img_anchor, vmax = 1, vmin = 0)
        self.ax_img_anchor.set_title(title)


    def plot_img_pos(self, title = ""): 
        self.ax_img_pos.imshow(self.img_pos, vmax = 1, vmin = 0)
        self.ax_img_pos.set_title(title)


    def plot_img_neg(self, title = ""): 
        self.ax_img_neg.imshow(self.img_neg, vmax = 1, vmin = 0)
        self.ax_img_neg.set_title(title)


    def plot_emb_anchor(self, title = ""): 
        self.ax_emb_anchor.plot(self.emb_anchor)
        ## self.ax_emb_anchor.set_aspect(15)
        self.ax_emb_anchor.set_title(title)


    def plot_emb_pos(self, title = ""): 
        self.ax_emb_pos.plot(self.emb_pos)
        ## self.ax_emb_pos.set_aspect(15)
        self.ax_emb_pos.set_title(title)


    def plot_emb_neg(self, title = ""): 
        self.ax_emb_neg.plot(self.emb_neg)
        ## self.ax_emb_neg.set_aspect(15)
        self.ax_emb_neg.set_title(title)


    def show(self): 
        self.fig.suptitle(self.title_fig)
        self.plot_img_anchor(self.title_anchor)
        self.plot_img_pos(self.title_pos)
        self.plot_img_neg(self.title_neg)
        self.plot_emb_anchor(self.title_anchor)
        self.plot_emb_pos(self.title_pos)
        self.plot_emb_neg(self.title_neg)
        plt.show()


## timestamp = "20220201224100"
## timestamp = "20220201224358"
## timestamp = "20220202145638"
## timestamp = "20220202184307"
## timestamp = "20220205133931"
## timestamp = "20220207114826"
timestamp = "20220207115721"

def init_weights(module):
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean = 0.5, std = 0.02)

# Config the dataset...
resize_y, resize_x = 6, 6
config_dataset = ConfigDataset( fl_csv         = 'datasets.csv',
                                size_sample    = 1000, 
                                resize         = (resize_y, resize_x),
                                ## exclude_labels = [ ConfigDataset.NOHIT, ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ],
                                exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ], 
                                isflat         = False, 
                              )
dataset_validate = SiameseDataset(config_dataset)

# Get image size...
spiimg = SPIImgDataset(config_dataset)
size_y, size_x = spiimg.get_img_and_label(0)[0].shape

# Config the encoder...
## dim_emb = 2
dim_emb = 10
config_encoder = ConfigEncoder( dim_emb = dim_emb,
                                size_y  = size_y,
                                size_x  = size_x,
                                isbias  = True )
encoder = Hirotaka0122(config_encoder)

# Load siamese model with random weights
alpha   = 1.0
config_siamese = ConfigSiameseModel( alpha = alpha, encoder = encoder, )
model_raw = SiameseModel(config_siamese)
model_raw.apply(init_weights)


# Load siamese model
model = SiameseModel(config_siamese)

drc_cwd = os.getcwd()
DRCCHKPT = "chkpts"
prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
if not os.path.exists(prefixpath_chkpt): os.makedirs(prefixpath_chkpt)

fl_chkpt = f"{timestamp}.train.chkpt"
path_chkpt = os.path.join(prefixpath_chkpt, fl_chkpt)
config_validator = ConfigValidator( path_chkpt  = path_chkpt,
                                    num_workers = 1,    # Visualization purpose
                                    batch_size  = 1,
                                    max_epochs  = 1,
                                    lr          = 0.0001, )

device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.current_device()

    model.load_state_dict(torch.load(config_validator.path_chkpt))
    model = torch.nn.DataParallel(model).to(device)


# Validate each epoch
for epoch in tqdm.tqdm(range(config_validator.max_epochs)):
    loader_validator = DataLoader( dataset_validate, shuffle     = True, 
                                                     pin_memory  = True, 
                                                     batch_size  = config_validator.batch_size,
                                                     num_workers = config_validator.num_workers )
    losses = []

    # Validate each batch
    batch = tqdm.tqdm(enumerate(loader_validator), total = len(loader_validator))
    for step_id, entry in batch:
        img_anchor, img_pos, img_neg, label_anchor, \
        title_anchor, title_pos, title_neg = entry

        ## print(title_anchor)

        for i in range(len(label_anchor)):
            emb_anchor, emb_pos, emb_neg, loss = model.forward(img_anchor[i].unsqueeze(0), img_pos[i].unsqueeze(0), img_neg[i].unsqueeze(0))
            viz_metadata = VizMetadata(size_y = size_y, size_x = size_x, figsize = (8*3,6*3),
                                       title_anchor = title_anchor[i],
                                       title_pos    = title_pos[i],
                                       title_neg    = title_neg[i],
                                       title_fig    = loss, )
            viz = VizSiameseOutput(img_anchor[i], img_pos[i], img_neg[i], emb_anchor[0], emb_pos[0], emb_neg[0], viz_metadata)
            viz.show()
