#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class SPIImgEmbed(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.conv(x)

        return x


class TripletLoss(nn.Module):
    """ Embedding independent triplet loss. """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha


    def forward(self, img_anchor_embed, img_pos_embed, img_neg_embed):
        ## # Encode images
        ## img_anchor_embed = self.encode(img_anchor)
        ## img_pos_embed    = self.encode(img_pos)
        ## img_neg_embed    = self.encode(img_neg)

        # Calculate the RMSD between anchor and positive
        img_diff = img_anchor_embed - img_pos_embed
        rmsd_anchor_pos = torch.sqrt( torch.mean(img_diff * img_diff) )

        # Calculate the RMSD between anchor and negative
        img_diff = img_anchor_embed - img_neg_embed
        rmsd_anchor_neg = torch.sqrt( torch.mean(img_diff * img_diff) )

        # Calculate the triplet loss
        loss_triplet = torch.max(rmsd_anchor_pos - rmsd_anchor_neg + self.alpha, 0)

        return loss_triplet.mean()
