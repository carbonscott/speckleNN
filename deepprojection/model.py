#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class SPIImageEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.conv(x)

        return x


class Siamese(SPIImageEmbedding):
    def __init__(self):
        super().__init__()

    def forward(self, img1, img2):
        img1_encoded = self.encode(img1)
        img2_encoded = self.encode(img2)

        # Calculate the root mean square error (RMSE) between two images
        img_diff = img1_encoded - img2_encoded
        img_rmse = torch.sqrt( torch.mean(img_diff * img_diff) )

        return img_rmse
