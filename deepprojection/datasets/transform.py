#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
import torch.nn as nn


class RandomPatch:
    """ Randomly place num_patch patch with the size of size_y * size_x onto an image.
    """

    def __init__(self, num_patch, size_patch_y,    size_patch_x, 
                                  var_patch_y = 0, var_patch_x = 0, 
                                  is_random_flip = False,
                                  is_return_mask = False):
        self.num_patch      = num_patch
        self.size_patch_y   = size_patch_y
        self.size_patch_x   = size_patch_x
        self.var_patch_y    = var_patch_y
        self.var_patch_x    = var_patch_x
        self.is_random_flip = is_random_flip
        self.is_return_mask = is_return_mask


    def __call__(self, img):
        # Get the size of the image...
        size_img_y, size_img_x = img.shape

        # Construct a mask of ones with the same size of the image...
        mask = np.ones_like(img)

        # Generate a number of random position...
        pos_y = np.random.randint(low = 0, high = size_img_y, size = self.num_patch)
        pos_x = np.random.randint(low = 0, high = size_img_x, size = self.num_patch)

        # Stack two column vectors to form an array of (x, y) indices...
        pos_y = pos_y.reshape(-1,1)
        pos_x = pos_x.reshape(-1,1)
        pos   = np.hstack((pos_y, pos_x))

        # Place patch of zeros at all pos as top-left corner...
        for (y, x) in pos:
            size_patch_y = self.size_patch_y
            size_patch_x = self.size_patch_x

            # Apply random variance...
            # Find the absolute max pixel to vary
            varsize_patch_y = int(size_patch_y * self.var_patch_y)
            varsize_patch_x = int(size_patch_x * self.var_patch_x)

            # Sample an integer from the min-max pixel to vary
            delta_patch_y = np.random.randint(low = -varsize_patch_y, high = varsize_patch_y)
            delta_patch_x = np.random.randint(low = -varsize_patch_x, high = varsize_patch_x)

            # Apply the change
            size_patch_y += delta_patch_y
            size_patch_x += delta_patch_x

            # Apply random flip???
            is_flip = random.choice((True, False)) if self.is_random_flip else self.is_random_flip
            if is_flip: size_patch_y, size_patch_x = size_patch_x, size_patch_y 

            # Find the limit of the bottom/right-end of the patch...
            y_end = min(y + size_patch_y, size_img_y)
            x_end = min(x + size_patch_x, size_img_x)

            # Patch the area with zeros...
            mask[y : y_end, x : x_end] = 0

        # Appy the mask...
        img *= mask

        # Construct the return value...
        output = img if not self.is_return_mask else img, mask

        return output
