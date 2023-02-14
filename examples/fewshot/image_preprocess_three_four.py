#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
from deepprojection.datasets import transform
from deepprojection.utils    import downsample, set_seed
import logging

logger = logging.getLogger(__name__)

set_seed(0)

class DatasetPreprocess:

    def __init__(self, img): 
        self.img = img

        # Set up state variable...
        self._counter_broken_panel = 0

        logger.info(f"___/ Preprocess Settings \___")

        return None


    def apply_mask(self):
        img = self.img
        size_y, size_x = img.shape

        # Create a raw mask...
        mask = np.zeros_like(img)

        ## # Mask out the top 10%...
        ## top = 0.1
        ## h_false = int(top * size_y)
        ## mask_false_area = (slice(0, h_false), slice(0, size_x))
        ## mask[mask_false_area[0], mask_false_area[1]] = 0

        # Mask out the oversaturated panel in 102...
        ## mask_false_area = (slice(510, None), slice(541, 670))
        mask_false_area = (slice(0, 510), slice(0, 541))
        mask[mask_false_area[0], mask_false_area[1]] = 1

        # Fetch the value...
        self.mask = mask

        logger.info(f"TRANS : Apply mask.")

        return None


    def apply_crop(self):
        crop_orig = 250 + 80, 250 + 80
        crop_end  = 250 + 550 - 80, 250 + 550 - 80

        trans_crop = transform.Crop(crop_orig, crop_end)

        self.trans_crop = trans_crop

        logger.info(f"TRANS : Apply cropping.")

        return None


    def apply_random_rotation(self):
        # Random rotation...
        angle = None
        center = (32, 32)
        ## center = (197, 197)
        ## center = (532, 527)
        trans_random_rotate = transform.RandomRotate(angle = angle, center = center)

        # Add augmentation to dataset configuration...
        self.trans_random_rotate = trans_random_rotate

        logger.info(f"TRANS : Apply random rotation. angle = {angle}, center = {center}")

        return None


    def apply_random_patch(self):
        # Random patching...
        num_patch = 10
        size_patch_y, size_patch_x = 10, 10
        trans_random_patch  = transform.RandomPatch(num_patch             , size_patch_y     , size_patch_x, 
                                                    var_patch_y    = 0.2  , var_patch_x    = 0.2, 
                                                    is_return_mask = False, is_random_flip = True)

        # Add augmentation to dataset configuration...
        self.trans_random_patch = trans_random_patch

        logger.info(f"TRANS : Apply random patching. size_patch_y = {size_patch_y}, size_patch_x = {size_patch_x}")

        return None


    def apply_downsample(self):
        resize_y, resize_x = 6, 6
        ## resize_y, resize_x = 12, 12
        resize = (resize_y, resize_x) if not None in (resize_y, resize_x) else ()

        self.resize = resize

        logger.info(f"TRANS : Apply downsampling. resize_y = {resize_y}, resize_x = {resize_x}")

        return None


    def apply_threshold(self):
        self.threshold_ok = True

        return None


    def apply_broken_half_panel(self, img):
        size_y, size_x = img.shape
        img[:size_x//2] = 0.0

        if not self._counter_broken_panel:
            logger.info(f"TRANS : Apply broken panel.  1/2 panel.")
            self._counter_broken_panel += 1

        return None


    def apply_broken_quater_panel(self, img):
        size_y, size_x = img.shape
        img[:size_x//2,:] = 0.0
        img[:, :size_y//2] = 0.0

        if not self._counter_broken_panel:
            logger.info(f"TRANS : Apply broken panel.  1/4 panel.")
            self._counter_broken_panel += 1

        return None


    def apply_broken_three_quater_panel(self, img):
        size_y, size_x = img.shape
        img[:size_x//2,:size_y//2] = 0.0
        img[:,size_y//2 + 1:]      = 0.0

        if not self._counter_broken_panel:
            logger.info(f"TRANS : Apply broken panel.  1/4 panel.")
            self._counter_broken_panel += 1

        return None


    def trans_threshold(self, img, threshold):
        img_norm = (img - img.mean()) / img.std()
        img[img_norm <= threshold] = 0.0

        logger.info(f"TRANS : Apply thresholding on normalized image. threshold = {threshold}")

        return None


    def trans(self, img, **kwargs):
        # Apply mask...
        if getattr(self, "mask", None) is not None: img *= self.mask

        # Apply crop...
        if getattr(self, "trans_crop", None) is not None: img = self.trans_crop(img)

        # Apply threshold...
        if getattr(self, "threshold_ok", None) is not None: 
            self.trans_threshold(img, threshold = 0.0)

        # Resize images...
        if getattr(self, "resize", None) is not None:
            bin_row, bin_col = self.resize
            img = downsample(img, bin_row, bin_col, mask = None)

        # Apply broken panel if available???
        if getattr(self, "is_broken_panel", None) is not None:
            ## self.apply_broken_half_panel(img)
            self.apply_broken_quater_panel(img)
            ## self.apply_broken_three_quater_panel(img)

        # Apply random rotation if available???
        if getattr(self, "trans_random_rotate", None) is not None:
            img = self.trans_random_rotate(img)

        # Apply random patching if available???
        if getattr(self, "trans_random_patch", None) is not None:
            img = self.trans_random_patch(img)

        return img


    def config_trans(self):
        ## self.apply_mask()
        self.apply_crop()
        ## self.apply_threshold()
        self.apply_downsample()
        self.apply_random_rotation()
        ## self.apply_random_patch()

        self.is_broken_panel = True

        return self.trans
        ## return None


    def save_random_state(self):
        self.state_random = random.getstate()

        return None


    def set_random_state(self):
        state_random = self.state_random
        random.setstate(state_random)

        return None
