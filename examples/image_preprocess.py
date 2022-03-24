#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from deepprojection.datasets.images import SPIImgDataset
from deepprojection.datasets        import transform
from deepprojection.utils           import downsample
import inspect

class DatasetPreprocess:

    def __init__(self, config_dataset): 
        self.config_dataset = config_dataset
        self.get_img()


    def get_img(self):
        config_dataset = self.config_dataset

        # Get image size...
        spiimg = SPIImgDataset(config_dataset)
        img    = spiimg.get_img_and_label(0)[0]

        self.spiimg = spiimg
        self.img    = img

        return None


    def get_imgsize(self): 
        self.get_img()

        return self.img.shape


    def apply_mask(self):
        img = self.img
        size_y, size_x = img.shape

        # Create a raw mask...
        mask = np.ones_like(img)

        # Mask out the top 10%...
        top = 0.1
        h_false = int(top * size_y)
        mask_false_area = (slice(0, h_false), slice(0, size_x))
        mask[mask_false_area[0], mask_false_area[1]] = 0

        # Mask out the oversaturated panel in 102...
        mask_false_area = (slice(510, None), slice(541, 670))
        mask[mask_false_area[0], mask_false_area[1]] = 0

        # Fetch the value...
        self.mask = mask

        return None


    def apply_augmentation(self):
        # Random rotation...
        angle = None
        center = (524, 506)
        trans_random_rotate = transform.RandomRotate(angle = angle, center = center)

        # Random patching...
        num_patch = 5
        size_patch_y, size_patch_x = 70, 500
        trans_random_patch  = transform.RandomPatch(num_patch             , size_patch_y     , size_patch_x, 
                                                    var_patch_y    = 0.2  , var_patch_x    = 0.2, 
                                                    is_return_mask = False, is_random_flip = True)
        trans_list = [trans_random_rotate, trans_random_patch]

        # Add augmentation to dataset configuration...
        self.trans_random = trans_list

        return None


    def apply_downsample(self):
        resize_y, resize_x = 6, 6
        resize = (resize_y, resize_x) if not None in (resize_y, resize_x) else ()

        self.resize = resize

        return None


    def trans(self, img, **kwargs):
        # Apply random transform if available???
        if getattr(self, "trans_random", None) is not None:
            for trans in self.trans_random:
                if isinstance(trans, (transform.RandomRotate, transform.RandomPatch)): img = trans(img)

        # Resize images...
        if getattr(self, "resize", None) is not None:
            bin_row, bin_col = self.resize
            img = downsample(img, bin_row, bin_col, mask = None)

        return img


    def apply(self):
        self.apply_mask()
        self.apply_augmentation()
        self.apply_downsample()

        self.config_dataset.trans = self.trans

        return None

