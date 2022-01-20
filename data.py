#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Load PyTorch
import torch
from torch.utils.data import Dataset, DataLoader

# Load LCLS data management and LCLS data access (through detector) module
import psana

# Load misc modules
import numpy as np
import json
import os


class SPIImageDataset(Dataset):
    """
    Single particle imaging (SPI) dataset.

    It loads SPI images and labels for machine learning with PyTorch, hence a
    data loader.  A label file, which contains the label of every image , is
    needed for a successful data loading.  
    """

    def __init__(self, exp, runs, mode, detector_name, drc_root):
        """
        Args:
            exp     (string)     : Experiment name, e.g. amo06516.
            runs    (list)       : Run numbers, e.g. [90, 91].
            mode    (string)     : Mode to load data.
            detector_name(string): Detector name, e.g. pnccdFront.
            drc_root(string)     : Root directory (drc) used in psocake processing.
        """
        # Load image labels to determine the total number of images
        self.num_imgs = 0
        self.labeled_event_dict = {}
        for run in runs:
            # Locate label file
            path_labelfile = self.locate_labelfile( exp, run, drc_root )

            # Read labels
            imglabel_dict = self.load_imglabel(path_labelfile)

            # Count number of labeled images in this run
            self.num_imgs += len(imglabel_dict)

            # Report to user
            print(f"Loading labeled events in run {run:04d}")

            # Save it
            self.labeled_event_dict[run] = self.get_labeled_event(imglabel_dict, exp, run, mode)

        # Initialize detector
        detector = psana.Detector(detector_name)

        # Load image data
        self.labels = []
        for i, (run, run_dict) in enumerate(self.labeled_event_dict.items()):
            for j, (event_num, event_dict) in enumerate(run_dict.items()):
                # Report to users
                print(f"Loading event {exp}.{run}.{event_num}")

                # Unique index of an image
                idx = i + j

                # Read image data from one event
                img = detector.image(event_dict["event"])

                # Preallocate memory for storing image data
                if idx == 0: 
                    self.size_y, self.size_x = img.shape
                    self.imgs = np.zeros((self.num_imgs, self.size_y * self.size_x))

                # Assign it to the img matrix
                self.imgs[idx] = img.reshape(-1)

                # Accumulate the label
                self.labels.append(event_dict["label"])


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, idx):
        img   = self.imgs[idx]
        label = self.labels[idx]

        return img, label


    def locate_labelfile(self, exp, run, drc_root):
        # Get the username
        username = os.environ.get("USER")

        # The prefix directory to find label file
        drc_run     = "r{run:04d}".format( run = run )
        drc_psocake = os.path.join(exp, username, 'psocake', drc_run)

        # Basename of a label file
        basename = f"{exp}_{run:04d}"

        # Locate the path to label file
        fl_json = f"{basename}.label.json"
        path_labelfile = os.path.join(drc_root, drc_psocake, fl_json)

        return path_labelfile


    def load_imglabel(self, path_labelfile):
        imglabel_dict = {}
        if os.path.exists(path_labelfile):
            with open(path_labelfile, 'r') as fh:
                imglabel_dict = json.load(fh)
            imglabel_dict = dict( sorted( imglabel_dict.items(), key = lambda x:int(x[0]) ) )
        else:
            print(f"File doesn't exist!!! Missing {path_labelfile}.")

        return imglabel_dict


    def get_labeled_event(self, imglabel_dict, exp, run, mode):
        # Biolerplate code to access an image
        datasource_id = f"exp={exp}:run={run}:{mode}"
        datasource    = psana.DataSource( datasource_id )
        run_current   = next(datasource.runs())
        timestamps    = run_current.times()

        # Read labeld images
        labeled_event_dict = {}
        for id, (event_num, label) in enumerate(imglabel_dict.items()):
            # Fetch the timestamp according to event number
            timestamp = timestamps[int(event_num)]

            # Access each event based on timestamp
            event = run_current.event(timestamp)

            # Store event in dictionary
            k = f"{int(event_num):04d}"
            labeled_event_dict[k] = { "event" : event, "label" : label }

        return labeled_event_dict




# Specify where the spi data are processed by psocake
drc_root = '/reg/data/ana03/scratch/cwang31/scratch/spi'

# Specify experiment and run for psana to accesss images
exp, run, mode = 'amo06516', [90, 91], 'idx'
detector_name = "pnccdFront"
dataset = SPIImageDataset(exp, run, mode, detector_name, drc_root)
