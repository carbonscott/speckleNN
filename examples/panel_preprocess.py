#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from deepprojection.datasets.panels import SPIPanelDataset
from deepprojection.datasets        import transform
from deepprojection.utils           import downsample
import inspect

class DatasetPreprocess:

    def __init__(self, config_dataset): 
        self.config_dataset = config_dataset
        self.get_panel()


    def get_panel(self):
        config_dataset = self.config_dataset

        # Get panel size...
        spipanel = SPIPanelDataset(config_dataset)
        panel = spipanel.get_panel_and_label(0)[0]

        self.spipanel = spipanel
        self.panel    = panel

        return None


    def get_panelsize(self):
        self.get_panel()

        return self.panel.shape


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
        ## # Random rotation...
        ## angle = None
        ## center = (524, 506)
        ## trans_random_rotate = transform.RandomRotate(angle = angle, center = center)

        # Random patching...
        num_patch = 5
        ## size_patch_y, size_patch_x = 40, 200
        size_patch_y, size_patch_x = 20, 80
        trans_random_patch  = transform.RandomPatch(num_patch             , size_patch_y     , size_patch_x, 
                                                    var_patch_y    = 0.2  , var_patch_x    = 0.2, 
                                                    is_return_mask = False, is_random_flip = True)
        ## trans_list = [trans_random_rotate, trans_random_patch]
        trans_list = [trans_random_patch]

        # Add augmentation to dataset configuration...
        self.trans_random = trans_list

        return None


    def apply_crop(self):
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
        self.trans_standardize = { 1 : transform.hflip, 3 : transform.hflip }

        return None


    def trans(self, img, **kwargs):
        """ The function consumed by dataset class.  
        """
        # Apply mask...
        if getattr(self, "mask", None) is not None: img *= self.mask

        # Apply transformation for standardizing image: low-right corner is the center...
        id_panel = kwargs.get("id_panel")
        if getattr(self, "trans_standardize", None) is not None:
            if id_panel in self.trans_standardize:
                trans = self.trans_standardize[id_panel]
                if inspect.isfunction(trans): img = trans(img)

        # Apply random transform if available???
        if getattr(self, "trans_random", None) is not None:
            for trans in self.trans_random:
                if isinstance(trans, (transform.RandomRotate, transform.RandomPatch)): img = trans(img)

        # Apply crop...
        if getattr(self, "trans_crop", None) is not None: img = self.trans_crop(img)

        # Apply zoom...
        if getattr(self, "trans_zoom", None) is not None: img = self.trans_zoom(img)
        if getattr(self, "RandomPanelZoom", None) is not None: img = self.trans_zoom(img)

        # Resize images...
        if getattr(self, "resize", None) is not None:
            bin_row, bin_col = self.resize
            img = downsample(img, bin_row, bin_col, mask = None)

        return img


    def apply(self):
        self.apply_mask()
        self.apply_standardize()
        ## self.apply_augmentation()
        self.apply_crop()
        self.apply_downsample()

        self.config_dataset.trans = self.trans

        return None
