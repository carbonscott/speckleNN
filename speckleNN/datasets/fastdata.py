#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import psana
import numpy as np
import random
import csv
import os
import logging

from torch.utils.data     import Dataset
from ..utils import set_seed, split_dataset

logger = logging.getLogger(__name__)

class ConfigDataset:
    ''' Biolerplate code to config dataset classs'''
    NOLABEL    = '-1'
    NOHIT      = '0'
    SINGLE     = '1'
    MULTI      = '2'
    UNKNOWN    = '3'
    NEEDHELP   = '4'
    BACKGROUND = '9'

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items(): setattr(self, k, v)


    def report(self):
        logger.info(f"___/ Configure Dataset \___")
        for k, v in self.__dict__.items():
            if k == 'kwargs': continue
            logger.info(f"KV - {k:16s} : {v}")




class SPIImgDataset(Dataset):
    """
    SPI images are collected from multiple datasets specified in the input csv
    file. All images are organized in a plain list.  
    """

    def __init__(self, config):
        fl_csv             = getattr(config, 'fl_csv'        , None)
        exclude_labels     = getattr(config, 'exclude_labels', None)
        self.mode          = getattr(config, 'mode'          , None)
        self.seed          = getattr(config, 'seed'          , None)
        self.trans         = getattr(config, 'trans'         , None)

        self._dataset_dict        = {}
        self.psana_imgreader_dict = {}
        self.imglabel_orig_list   = []
        self.imglabel_cache_dict  = {}
        self.is_cache             = False

        # Set the seed...
        # Debatable whether seed should be set in the dataset or in the running code
        if not self.seed is None: set_seed(self.seed)

        # Read csv file of datasets...
        with open(fl_csv, 'r') as fh:
            lines = csv.reader(fh)

            # Skip the header...
            next(lines)

            # Read each line/dataset...
            for line in lines:
                # Fetch metadata of a dataset 
                exp, run, mode, detector_name, drc_label = line

                # Form a minimal basename to describe a dataset...
                basename = (exp, run)

                # Initiate image accessing layer...
                psana_imgreader = PsanaImg(exp, run, mode, detector_name)
                self.psana_imgreader_dict[basename] = psana_imgreader

                # Obtain image labels from this dataset...
                imglabel_fileparser = ImgLabelFileParser(exp, run, drc_label, exclude_labels)

                # Parse labels in the label file if it exists???
                self._dataset_dict[basename] = imglabel_fileparser.imglabel_dict

        # Enumerate each labeled image from all datasets...
        for dataset_id, dataset_content in self._dataset_dict.items():
            # Get the exp and run
            exp, run = dataset_id

            for event_num, label in dataset_content.items():
                self.imglabel_orig_list.append( (exp, run, f"{event_num:>6s}", label) )

        self.imglabel_list = self.imglabel_orig_list

        return None


    def __len__(self):
        return len(self.imglabel_list)


    def cache_img(self, idx_list = []):
        ''' Cache the whole dataset in imglabel_list or some subset.
        '''
        # If subset is not give, then go through the whole set...
        if not len(idx_list): idx_list = range(len(self.imglabel_list))

        for idx in idx_list:
            # Skip those have been recorded...
            if idx in self.imglabel_cache_dict: continue

            # Otherwise, record it
            img, label = self.get_img_and_label(idx, verbose = True)
            self.imglabel_cache_dict[idx] = (img, label)

        return None


    def get_img_and_label(self, idx, verbose = False):
        # Read image...
        exp, run, event_num, label = self.imglabel_list[idx]
        basename = (exp, run)
        img = self.psana_imgreader_dict[basename].get(int(event_num), mode = self.mode)

        if verbose: logger.info(f'DATA LOADING - {exp} {run} {event_num} {label}.')

        return img, label


    def __getitem__(self, idx):
        img, label = self.imglabel_cache_dict[idx] if   idx in self.imglabel_cache_dict \
                                                   else self.get_img_and_label(idx)

        # Apply any possible transformation...
        # How to define a custom transform function?
        # Input : img, **kwargs 
        # Output: img_transfromed
        if self.trans is not None:
            img = self.trans(img)

        ## # Normalize input image...
        ## img_mean = np.mean(img)
        ## img_std  = np.std(img)
        ## img_norm = (img - img_mean) / img_std

        img_norm = img

        return img_norm, int(label), self.imglabel_list[idx]




class ImgLabelFileParser:
    """
    It parses a label file associated with a run in an experiment.  The label 
    file, a csv file of event number and labels, should be generated by
    psocake.  This parser numerically sorts the event number and assign a
    zero-based index to each event number.  This is implemented primarily for
    complying with PyTorch DataLoader.  

    The type of a lable is always string.  
    """

    def __init__(self, exp, run, drc_label, exclude_labels = ()):
        self.exp            = exp
        self.run            = run
        self.drc_label      = drc_label
        self.path_labelfile = ""
        self.imglabel_dict  = {}
        self.exclude_labels = exclude_labels

        # Initialize indexed image label
        self._load_imglabel()

        return None


    def __getitem__(self, idx):
        return self.indexed_imglabel_dict[idx]


    def _locate_labelfile(self):
        # Basename of a label file
        basename = f"{self.exp}_{int(self.run):04d}"

        # Locate the path to label file
        fl_label = f"{basename}.label.csv"
        path_labelfile = os.path.join(self.drc_label, fl_label)

        return path_labelfile


    def _load_imglabel(self):
        # Load path to the label file
        self.path_labelfile = self._locate_labelfile()

        # Read, sort and index labels
        imglabel_dict = {}
        if os.path.exists(self.path_labelfile):
            # Read csv file of datasets
            with open(self.path_labelfile, 'r') as fh:
                lines = csv.reader(fh)

                # Skip the header
                next(lines)

                # Read each line/dataset
                for line in lines:
                    # Fetch metadata of a dataset 
                    _, _, event_num, label = line
                    imglabel_dict[event_num] = label

            # Exclude some labels
            imglabel_dict = { k:v for k, v in imglabel_dict.items() if not v in self.exclude_labels }

            # Sort label
            self.imglabel_dict = dict( sorted( imglabel_dict.items(), key = lambda x:int(x[0]) ) )

        return None




class PsanaImg:
    """
    It serves as an image accessing layer based on the data management system
    psana in LCLS.  
    """

    def __init__(self, exp, run, mode, detector_name):
        # Biolerplate code to access an image
        # Set up data source
        self.datasource_id = f"exp={exp}:run={run}:{mode}"
        self.datasource    = psana.DataSource( self.datasource_id )
        self.run_current   = next(self.datasource.runs())
        self.timestamps    = self.run_current.times()

        # Set up detector
        self.detector = psana.Detector(detector_name)


    def get(self, event_num, mode = "image"):
        # Fetch the timestamp according to event number...
        timestamp = self.timestamps[int(event_num)]

        # Access each event based on timestamp...
        event = self.run_current.event(timestamp)

        # Only three modes are supported...
        assert mode in ("raw", "image", "calib"), f"Mode {mode} is not allowed!!!  Only 'raw' or 'image' are supported."

        # Fetch image data based on timestamp from detector...
        read = { "image" : self.detector.image, }
        img = read[mode](event)

        return img
