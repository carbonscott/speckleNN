#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import psana

class PsanaImg:
    """
    It serves as a shortcut to access an detector images at LCLS.  `psana` does
    all the heaviy lifting under the hood.  For more information about `psana`, 
    please go to 

    - https://confluence.slac.stanford.edu/display/PSDM/Jump+Quickly+to+Events+Using+Timestamps
    - https://confluence.slac.stanford.edu/display/PSDM/LCLS+Data+Analysis
    """

    def __init__(self, exp, run, mode, detector_name):
        """
        Parameters
        ----------
        exp : str
            Experiment names like 'amo06516'.

        run : str or int
            Run number inside an experiment.

        mode : str
            Specify how to fetch data.

        detector_name : str
            Detector name.

        Examples
        --------
        >>> PsanaImg(exp = 'amo06516', run = 101, mode = 'idx', detector_name = 'pnccdFront')

        """
        # Biolerplate code to access an image
        # Set up data source
        self.datasource_id = f"exp={exp}:run={run}:{mode}"
        self.datasource    = psana.DataSource( self.datasource_id )
        self.run_current   = next(self.datasource.runs())
        self.timestamps    = self.run_current.times()

        # Set up detector
        self.detector = psana.Detector(detector_name)


    def get(self, event_num, multipanel = None, mode = "image"):
        """
        Return detector image data.  
        """
        # Fetch the timestamp according to event number...
        timestamp = self.timestamps[int(event_num)]

        # Access each event based on timestamp...
        event = self.run_current.event(timestamp)

        # Only three modes are supported...
        assert mode in ("raw", "image", "calib"), f"Mode {mode} is not allowed!!!  Only 'raw' or 'image' are supported."

        # Fetch image data based on timestamp from detector...
        read = { "image" : self.detector.image,
                 "calib" : self.detector.calib }
        img = read[mode](event) if multipanel is None else read[mode](event, multipanel)

        return img




class DatasetModifierForModelShi(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset


    def __getitem__(self, idx):
        data, label, metadata = self.dataset[idx]

        if label != 1: label = 0

        return data, label, metadata


    def __len__(self):
        return len(self.dataset)




class DatasetModifierForSkopiData(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset


    def __getitem__(self, idx):
        data, label, metadata = self.dataset[idx]

        if label != 1: label = 2

        return data, label, metadata


    def __len__(self):
        return len(self.dataset)
