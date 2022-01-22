#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader


logger = logging.getLogger(__name__)

class TrainerConfig:
    checkpoint_path = None
    num_workers     = 0
    batch_size      = 64
    max_epochs      = 10
    alpha           = 0.1
    lr              = 0.001

    def __init__(self, **kwargs):
        # Set values of attributes that are not known when obj is created
        for k, v in kwargs.items():
            setattr(self, k, v)


class TripletLoss(nn.Module):
    """ Embedding independent triplet loss. """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha


    def forward(self, img_anchor_embed, img_pos_embed, img_neg_embed):
        # Calculate the RMSD between anchor and positive
        img_diff = img_anchor_embed - img_pos_embed
        rmsd_anchor_pos = torch.sqrt( torch.mean(img_diff * img_diff) )

        # Calculate the RMSD between anchor and negative
        img_diff = img_anchor_embed - img_neg_embed
        rmsd_anchor_neg = torch.sqrt( torch.mean(img_diff * img_diff) )

        # Calculate the triplet loss
        loss_triplet = torch.max(rmsd_anchor_pos - rmsd_anchor_neg + self.alpha, 0)

        return loss_triplet


class Trainer:
    def __init__(self, model, dataset_train, dataset_test, config):
        self.model         = model
        self.dataset_train = dataset_train
        self.dataset_test  = dataset_test
        self.config        = config

        # Load data to gpus if available
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model  = torch.nn.DataParallel(self.model).to(self.device)

        return None


    def save_checkpoint(self):
        model_raw = self.model.module if hasattr(self.model, "module") else self.model
        logger.info(f"Saving {self.config.checkpoint_path}")
        torch.save(model_raw.state_dict(), self.config.checkpoint_path)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config.lr)

        return optimizer


    def train(self):
        model, config = self.model, self.config
        model_raw = model.module if hasattr(model, "module") else model
        optimizer = self.configure_optimizers()

        tripletloss = torch.jit.script(TripletLoss(config.alpha))

        for epoch in tqdm(range(config.max_epochs)):
            model.train()
            dataset_train = self.dataset_train
            loader_train = DataLoader( dataset_train, shuffle = True, 
                                                      pin_memory = True, 
                                                      batch_size = config.batch_size,
                                                      num_workers = config.num_workers )
            losses = []
            progbar = tqdm(enumerate(loader_train), total = len(loader_train))
            for step_id, (img_anchor, img_pos, img_neg, label_anchor) in progbar:
                img_anchor = img_anchor.to(self.device)
                img_pos    = img_pos.to(self.device)
                img_neg    = img_neg.to(self.device)

                optimizer.zero_grad()
                img_anchor_embed = model(img_anchor)
                img_pos_embed    = model(img_pos)
                img_neg_embed    = model(img_neg)

                loss = tripletloss(img_anchor_embed, img_pos_embed, img_neg_embed)
                loss.backward()
                optimizer.step()

                losses.append(loss.cpu().detach().numpy())

            print(f"Epoch: {epoch + 1}/{config.max_epochs} - Loss: {np.mean(loss):.4f}")
