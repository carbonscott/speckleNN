#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from deepprojection.dataset import SPIImageDataset
from deepprojection.model import Siamese, SPIImageEmbedding
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

    ## def label_panels():
    ##     ax_img1.set_ylabel("")


fl_csv = 'datasets.csv'

img_dataset = SPIImageDataset(fl_csv)
dataloader  = DataLoader(img_dataset, batch_size = 2, num_workers = 1)

for i, (img_batch, label_batch) in enumerate(dataloader):
    img_batch = img_batch.to('cuda')
    label_batch = torch.as_tensor(label_batch).to('cuda')
    print(f"Process batch {i} with {len(img_batch)} images.")

    break


size_y, size_x = img_dataset.get_imagesize(0)
x1, x2 = img_batch[0], img_batch[1]

## net = Siamese()
## x_diff = net.forward(x1, x2)

spi_emb = SPIImageEmbedding()
x1_embedded = spi_emb.encode(x1)

img1 = x1.reshape(size_y, size_x).cpu().numpy()
img2 = x1_embedded.reshape(size_y, size_x).cpu().numpy()

## plt.imshow(x1.reshape(size_y, size_x).cpu().numpy(), vmax = 100, vmin = 0)

## plt.imshow(x1_embedded.reshape(size_y, size_x).cpu().numpy(), vmax = 1.0, vmin = 0.8)

imgcmp = ImageCompare(img1, img2)
imgcmp.plot_img1(title = "Original")
imgcmp.plot_img2(title = "Embedding")
imgcmp.show()
