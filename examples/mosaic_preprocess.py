#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from deepprojection.datasets.mosaic import SPIMosaicDataset
from deepprojection.datasets        import transform
from deepprojection.utils           import downsample

class DatasetPreprocess:

    def __init__(self, config_dataset): 
        self.config_dataset = config_dataset
        self.get_panel()
        self.get_mosaic()


    def get_panel(self):
        config_dataset = self.config_dataset

        # Get panel size...
        spipanel = SPIMosaicDataset(config_dataset)
        spipanel.IS_MOSAIC = False
        panels, _ = spipanel.get_img_and_label(0)

        panel = panels[0]

        self.panel = panel

        return None


    def get_mosaic(self):
        config_dataset = self.config_dataset

        # Get panel size...
        spipanel = SPIMosaicDataset(config_dataset)
        spipanel.IS_MOSAIC = True
        img_mosaic, _ = spipanel.get_img_and_label(0)

        self.img_mosaic = img_mosaic

        return None


    def get_panelsize(self): 
        self.get_panel()

        return self.panel.shape


    def get_mosaicsize(self):
        self.get_mosaic()

        return self.img_mosaic.shape


    def apply_mask(self):
        panel = self.panel
        size_y, size_x = panel.shape

        # Create a raw mask...
        mask = np.ones_like(panel)

        # Mask out the top 20%...
        top = 0.2
        h_false = int(top * size_y)
        mask_false_area = (slice(0, h_false), slice(0, size_x))
        mask[mask_false_area[0], mask_false_area[1]] = 0

        # Fetch the value...
        self.mask = mask

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

        return None


    def apply_crop(self):
        # [TO FIX]
        panel = self.panel
        crop_orig = 250, 250
        crop_end  = panel.shape

        trans_crop = transform.Crop(crop_orig, crop_end)

        self.trans_crop = trans_crop

        return None



    def apply_downsample(self):
        resize_y, resize_x = 6, 6
        resize = (resize_y, resize_x) if not None in (resize_y, resize_x) else ()

        self.resize = resize

        return None


    def apply_standardize(self):
        self.trans_standardize = { "1" : transform.hflip, "3" : transform.hflip }

        return None


    def trans(self, imgs, **kwargs):
        imgs_trans = []
        for img in imgs:
            # Apply mask...
            if getattr(self, "mask", None) is not None: img *= self.mask

            # Apply random transform if available???
            if getattr(self, "trans_random", None) is not None:
                for trans in self.trans_random:
                    if isinstance(trans, (transform.RandomRotate, transform.RandomPatch)): img = trans(img)

            # Apply crop...
            if getattr(self, "trans_crop", None) is not None: img = self.trans_crop(img)

            # Resize images...
            if getattr(self, "resize", None) is not None:
                bin_row, bin_col = self.resize
                img = downsample(img, bin_row, bin_col, mask = None)

            imgs_trans.append(img)

        return imgs_trans



    def apply(self):
        self.apply_mask()
        ## self.apply_standardize()
        ## self.apply_augmentation()
        ## self.apply_crop()
        self.apply_downsample()

        self.config_dataset.trans = self.trans

        return None
