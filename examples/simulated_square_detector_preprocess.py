#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import inspect
import logging
from deepprojection.datasets import transform
from deepprojection.utils    import downsample

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

        # Mask out the top 20%...
        top = 0.2
        h_false = int(top * size_y)
        mask_false_area = (slice(0, h_false), slice(0, size_x))
        mask[mask_false_area[0], mask_false_area[1]] = 0

        # Fetch the value...
        self.mask = mask

        logger.info(f"Apply mask.")

        return None


    def apply_augmentation(self):
        # Random rotation...
        angle = None
        center = (524, 506)
        trans_random_rotate = transform.RandomRotate(angle = angle, center = center)

        # Random patching...
        num_patch = 5
        size_patch_y, size_patch_x = 2, 40
        trans_random_patch  = transform.RandomPatch(num_patch             , size_patch_y     , size_patch_x, 
                                                    var_patch_y    = 0.2  , var_patch_x    = 0.2, 
                                                    is_return_mask = False, is_random_flip = True)
        ## trans_list = [trans_random_rotate, trans_random_patch]
        trans_list = [trans_random_patch]

        # Add augmentation to dataset configuration...
        self.trans_random = trans_list

        logger.info(f"Apply random patching.")

        return None


    def apply_crop(self):
        img = self.img
        crop_orig = 43 - 15, 43 - 15
        crop_end  = 43 + 15, 43 + 15

        trans_crop = transform.Crop(crop_orig, crop_end)

        self.trans_crop = trans_crop

        logger.info(f"Apply cropping. From {crop_orig[0]}, {crop_orig[1]} to {crop_end[0]}, {crop_end[1]}")

        return None



    def apply_downsample(self):
        resize_y, resize_x = 2, 2
        resize = (resize_y, resize_x) if not None in (resize_y, resize_x) else ()

        self.resize = resize

        logger.info(f"Apply downsampling.  Resize y by {resize_y} folds.  Resize x by {resize_x} folds.")

        return None


    def apply_standardize(self):
        self.trans_standardize = { 1 : transform.hflip, 3 : transform.hflip }

        logger.info(f"Apply standardization.")

        return None


    def apply_zoom(self):
        img = self.img
        size_y, size_x = img.shape

        ## low = 0
        ## high = int(0.6 * size_y)
        mid  = int(0.5 * size_y)
        low  = int(mid - 0.05 * size_y)
        high = int(mid + 0.05 * size_y)

        trans_zoom = transform.RandomPanelZoom(low = low, high = high)

        self.trans_zoom = trans_zoom

        logger.info(f"Apply random zooming. low = {low}, high = {high}.")

        return None


    def apply_noise_poisson(self):
        self.noise_poisson = transform.noise_poisson

        logger.info("Apply Poisson noise. ")

        return None


    def apply_noise_gaussian(self):
        scale = 10.0
        sigma = scale * 0.15

        def _noise_gaussian(img):
            return transform.noise_gaussian(img, sigma)

        self.noise_gaussian = _noise_gaussian

        logger.info(f"Apply Gaussian noise. sigma = {sigma}.")

        return None


    def trans(self, img, **kwargs):
        """ The function consumed by dataset class.  
        """
        # Apply mask...
        if getattr(self, "mask", None) is not None: img *= self.mask

        # Apply zoom...
        if getattr(self, "trans_zoom", None) is not None: img = self.trans_zoom(img)
        if getattr(self, "RandomPanelZoom", None) is not None: img = self.trans_zoom(img)

        # Resize images...
        if getattr(self, "resize", None) is not None:
            bin_row, bin_col = self.resize
            img = downsample(img, bin_row, bin_col, mask = None)

        # Apply crop...
        if getattr(self, "trans_crop", None) is not None: img = self.trans_crop(img)

        # Apply random transform if available???
        if getattr(self, "trans_random", None) is not None:
            for trans in self.trans_random:
                if isinstance(trans, (transform.RandomRotate, transform.RandomPatch)): img = trans(img)

        # Apply Poisson noise...
        if getattr(self, "noise_poisson", None) is not None:
            scale_factor = 1.0
            img = self.noise_poisson(img, scale_factor)

        # Apply Guassian noise...
        if getattr(self, "noise_gaussian", None) is not None:
            img = self.noise_gaussian(img)

        return img


    def config_trans(self):
        self.apply_noise_poisson()
        self.apply_noise_gaussian()

        ## self.apply_mask()
        self.apply_augmentation()
        self.apply_crop()

        self.apply_downsample()

        return self.trans
