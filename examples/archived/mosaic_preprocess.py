#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from deepprojection.datasets        import transform
from deepprojection.utils           import downsample
import logging

logger = logging.getLogger(__name__)


class DatasetPreprocess:

    def __init__(self, img, panels_ordered): 
        self.img            = img
        self.panels_ordered = panels_ordered    # Enable panel-based preprocessing

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
        # [TO FIX]
        ## # Random rotation...
        ## angle = None
        ## center = (524, 506)
        ## trans_random_rotate = transform.RandomRotate(angle = angle, center = center)

        # Random patching...
        num_patch = 2
        size_patch_y, size_patch_x = 40, 200
        trans_random_patch  = transform.RandomPatch(num_patch             , size_patch_y        , size_patch_x, 
                                                    var_patch_y    = 0.2  , var_patch_x    = 0.2, 
                                                    is_return_mask = False, is_random_flip = True)
        ## trans_list = [trans_random_rotate, trans_random_patch]
        trans_list = [trans_random_patch]

        # Add augmentation to dataset configuration...
        self.trans_random = trans_list

        logger.info(f"Apply random patching.")

        return None


    def apply_crop(self, idx_panel):
        panels_ordered = self.panels_ordered

        offset = 80
        crop_dict = {
            0 : { 'orig' : (250 + offset, 250 + offset), 'end' : (500, 500) },
            1 : { 'orig' : (250 + offset,   0),          'end' : (500, 250 - offset) },
            2 : { 'orig' : (250 + offset, 250 + offset), 'end' : (500, 500) },
            3 : { 'orig' : (250 + offset,   0),          'end' : (500, 250 - offset) },
        }

        crop_orig = crop_dict[idx_panel]['orig']
        crop_end  = crop_dict[idx_panel]['end']

        trans_crop = transform.Crop(crop_orig, crop_end)

        self.trans_crop = trans_crop

        logger.info(f"Apply cropping.")

        return None



    def apply_downsample(self):
        resize_y, resize_x = 6, 6
        resize = (resize_y, resize_x) if not None in (resize_y, resize_x) else ()

        self.resize = resize

        logger.info(f"Apply downsampling. resize_y = {resize_y}, resize_x = {resize_x}")

        return None


    def trans(self, imgs, **kwargs):
        imgs_trans = []
        for i, idx_panel in enumerate(self.panels_ordered):
            self.apply_crop(idx_panel)

            img = imgs[i]

            # Apply mask...
            if getattr(self, "mask", None) is not None: img *= self.mask

            # Apply crop...
            if getattr(self, "trans_crop", None) is not None: img = self.trans_crop(img)

            # Resize images...
            if getattr(self, "resize", None) is not None:
                bin_row, bin_col = self.resize
                img = downsample(img, bin_row, bin_col, mask = None)

            # Apply random transform if available???
            if getattr(self, "trans_random", None) is not None:
                for trans in self.trans_random:
                    if isinstance(trans, (transform.RandomRotate, transform.RandomPatch)): img = trans(img)

            imgs_trans.append(img)

        return imgs_trans



    def config_trans(self):
        ## self.apply_mask()
        ## self.apply_augmentation()
        ## self.apply_crop()
        self.apply_downsample()

        return self.trans
