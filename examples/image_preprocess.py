#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from deepprojection.datasets import transform
from deepprojection.utils    import downsample
import logging

logger = logging.getLogger(__name__)


class DatasetPreprocess:

    def __init__(self, img): 
        self.img = img

        logger.info(f"___/ Preprocess Settings \___")

        return None


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

        logger.info(f"Apply mask.")

        return None


    def apply_crop(self):
        crop_orig = 250 + 80, 250 + 80
        crop_end  = 250 + 550 - 80, 250 + 550 - 80

        trans_crop = transform.Crop(crop_orig, crop_end)

        self.trans_crop = trans_crop

        logger.info(f"Apply cropping.")

        return None


    def apply_augmentation(self):
        # Random rotation...
        angle = None
        center = (32, 32)
        ## center = (532, 527)
        trans_random_rotate = transform.RandomRotate(angle = angle, center = center)

        # Random patching...
        num_patch = 5
        ## size_patch_y, size_patch_x = 6, 50
        size_patch_y, size_patch_x = 6, 30
        trans_random_patch  = transform.RandomPatch(num_patch             , size_patch_y     , size_patch_x, 
                                                    var_patch_y    = 0.2  , var_patch_x    = 0.2, 
                                                    is_return_mask = False, is_random_flip = True)
        trans_list = [trans_random_rotate]
        ## trans_list = [trans_random_patch]
        ## trans_list = [trans_random_rotate, trans_random_patch]

        # Add augmentation to dataset configuration...
        self.trans_random = trans_list

        logger.info(f"Apply random patching. angle = {angle}, center = {center}, size_patch_y = {size_patch_y}, size_patch_x = {size_patch_x}")

        return None


    def apply_downsample(self):
        resize_y, resize_x = 6, 6
        ## resize_y, resize_x = 12, 12
        resize = (resize_y, resize_x) if not None in (resize_y, resize_x) else ()

        self.resize = resize

        logger.info(f"Apply downsampling. resize_y = {resize_y}, resize_x = {resize_x}")

        return None


    def apply_threshold(self):
        self.threshold_ok = True

        return None



    def trans_threshold(self, img, threshold):
        img_norm = (img - img.mean()) / img.std()
        img[img_norm <= threshold] = 0.0

        logger.info(f"Apply thresholding on normalized image. threshold = {threshold}")

        return img


    def trans(self, img, **kwargs):
        # Apply mask...
        if getattr(self, "mask", None) is not None: img *= self.mask

        # Apply crop...
        if getattr(self, "trans_crop", None) is not None: img = self.trans_crop(img)

        # Apply threshold...
        if getattr(self, "threshold_ok", None) is not None: 
            img = self.trans_threshold(img, threshold = 1)

        # Resize images...
        if getattr(self, "resize", None) is not None:
            bin_row, bin_col = self.resize
            img = downsample(img, bin_row, bin_col, mask = None)

        # Apply random transform if available???
        if getattr(self, "trans_random", None) is not None:
            for trans in self.trans_random:
                if isinstance(trans, (transform.RandomRotate)): img = trans(img)

        # Apply random transform if available???
        if getattr(self, "trans_random", None) is not None:
            for trans in self.trans_random:
                if isinstance(trans, (transform.RandomPatch)): img = trans(img)

        return img


    def config_trans(self):
        ## self.apply_mask()
        self.apply_crop()
        ## self.apply_threshold()
        self.apply_downsample()
        self.apply_augmentation()

        return self.trans
