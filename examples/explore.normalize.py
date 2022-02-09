#!/usr/bin/env python
# -*- coding: utf-8 -*-

import psana
import matplotlib.pyplot as plt
import numpy as np


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
        # Fetch the timestamp according to event number
        timestamp = self.timestamps[int(event_num)]

        # Access each event based on timestamp
        event = self.run_current.event(timestamp)

        read = { "image" : self.detector.image,
                 "raw"   : self.detector.raw,}
        # Fetch image data based on timestamp from detector
        ## img = self.detector.image(event)
        img = read[mode](event)

        return img




class HistSPIImg():

    def __init__(self, img, figsize, **kwargs):
        self.img = img
        self.figsize = figsize
        for k, v in kwargs.items(): setattr(self, k, v)

        self.img_mean = np.mean(img)
        self.img_std  = np.std(img)

        self.fig, (self.ax_img, self.ax_norm) = self.create_panels()


    def create_panels(self):
        plt.rcParams.update({'font.size': 18})
        plt.rcParams.update({'font.family' : 'sans-serif'})

        nrows, ncols = 1, 2
        fig = plt.figure(figsize = self.figsize)

        gspec   = fig.add_gridspec(nrows, ncols)
        ax_img  = fig.add_subplot(gspec[0,0])
        ax_norm = fig.add_subplot(gspec[0,1])
        return fig, (ax_img, ax_norm)


    def plot_img(self, rng = [], bin_cap = 10000): 
        img = self.img
        bin_val, bin_rng = self.population_density(img, bin_cap)

        img_mean = self.img_mean
        img_std  = self.img_std

        if len(rng) == 0: rng = bin_rng[0], bin_rng[-1]
        data = []
        for i in range(len(bin_val)):
            data.append([bin_rng[i], bin_val[i]])
            data.append([bin_rng[i+1], bin_val[i]])
        data = np.array(data)
        self.ax_img.plot(data[:, 0], data[:, 1])
        self.ax_img.set_yscale('log')
        self.ax_img.set_xlim(rng[0], rng[1])
        self.ax_img.set_ylim(min(bin_val), max(bin_val))
        self.ax_img.set_title(f"$\mu = {img_mean:.4f}$; $\sigma = {img_std:.4f};  b = {bin_cap}$")
        self.ax_img.set_xlabel('Pixel intensity')
        self.ax_img.set_ylabel('Population density')
        print(f"Image: ")
        print(f"Range X: [{bin_rng[0]}, {bin_rng[-1]}]")
        print(f"Range Y: [{bin_val[0]}, {bin_val[-1]}]")


    def plot_norm(self, rng = [], bin_cap = 10000): 
        img = self.img
        img_mean = self.img_mean
        img_std  = self.img_std

        img_norm = (img - img_mean) / img_std
        img_mean = np.mean(img_norm)
        img_std  = np.std(img_norm)
        bin_val, bin_rng = self.population_density(img_norm, bin_cap)

        if len(rng) == 0: rng = bin_rng[0], bin_rng[-1]
        data = []
        for i in range(len(bin_val)):
            data.append([bin_rng[i], bin_val[i]])
            data.append([bin_rng[i+1], bin_val[i]])
        data = np.array(data)
        self.ax_norm.plot(data[:, 0], data[:, 1])
        self.ax_norm.set_yscale('log')
        self.ax_norm.set_xlim(rng[0], rng[1])
        self.ax_norm.set_ylim(min(bin_val), max(bin_val))
        self.ax_norm.set_title(f"$\mu = {img_mean:.4f}$; $\sigma = {img_std:.4f};  b = {bin_cap}$")
        self.ax_norm.set_xlabel('Pixel intensity')
        self.ax_norm.set_ylabel('Population density')
        print(f"Norm: ")
        print(f"Range X: [{bin_rng[0]}, {bin_rng[-1]}]")
        print(f"Range Y: [{bin_val[0]}, {bin_val[-1]}]")


    def show(self): 
        self.plot_img (rng = [], bin_cap = 200)
        self.plot_norm(rng = [], bin_cap = 200)

        img_mean = self.img_mean
        img_std  = self.img_std
        ## plt.suptitle(f"mean = {img_mean:.4f}; std = {img_std:.4f}")
        plt.show()


    def population_density(self, data, bin_cap = 100):
        ''' Return population density.
            bin_cap stands for bin capacity (number of items per bin).
        '''
        # Flatten data...
        data_flat = data.reshape(-1)

        # Sort data...
        data_sort = np.sort(data_flat)

        # Obtain the length of data...
        s, = data_sort.shape

        # Go through the array and figure out bin_val and bin_edge...
        bin_val  = []
        bin_edge = []
        bin_step = bin_cap
        for i in range(0, s, bin_cap):
            if i + bin_cap > s: bin_step = s - i
            data_seg = data_sort[i : i + bin_step]
            b, e = data_seg[0], data_seg[-1]
            if abs(e - b) < 10e-6: 
                print(f"The bin cap (={bin_cap}) is too small, some population density reaches to infinity!!!")
                den = 1e-1
            else: den = bin_step / (e - b)
            bin_val.append(den)
            bin_edge.append(b)
        bin_edge.append( data_sort[-1] )

        return bin_val, bin_edge




# Specify the dataset and detector...
exp, run, mode, detector_name = 'amo06516', '90', 'idx', 'pnccdFront'

# Initialize an image reader...
img_reader = PsanaImg(exp, run, mode, detector_name)

# Access an image (e.g. event 796)...
event_num = 796
img = img_reader.get(event_num, mode = "image")
## img_raw = img_reader.get(event_num, mode = "raw")

# Dispaly an image...
disp_manager = HistSPIImg(img, figsize = (18, 8))
disp_manager.show()
