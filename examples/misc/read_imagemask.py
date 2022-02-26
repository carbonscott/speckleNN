#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import psana
import matplotlib              as mpl
import matplotlib.pyplot       as plt
import matplotlib.colors       as mcolors
import matplotlib.patches      as mpatches
import matplotlib.transforms   as mtransforms
import matplotlib.font_manager as font_manager
import numpy as np
from deepprojection.datasets import transform

class DisplaySPIImg():

    def __init__(self, img, mask, figsize, **kwargs):
        self.img = img
        self.mask = mask
        self.figsize = figsize
        for k, v in kwargs.items(): setattr(self, k, v)

        self.config_fonts()
        self.config_colorbar()

    def config_fonts(self):
        # Where to load external font...
        drc_py    = os.path.dirname(os.path.realpath(__file__))
        drc_font  = os.path.join("fonts", "Helvetica")
        fl_ttf    = f"Helvetica.ttf"
        path_font = os.path.join(drc_py, drc_font, fl_ttf)
        prop_font = font_manager.FontProperties( fname = path_font )

        # Add Font and configure font properties
        font_manager.fontManager.addfont(path_font)
        prop_font = font_manager.FontProperties(fname = path_font)
        self.prop_font = prop_font

        # Specify fonts for pyplot...
        plt.rcParams['font.family'] = prop_font.get_name()
        plt.rcParams['font.size']   = 18

        return None


    def create_panels(self):
        nrows, ncols = 1, 2
        fig = plt.figure(figsize = self.figsize)
        gspec =  fig.add_gridspec(nrows, ncols)
        ax_img  = fig.add_subplot(gspec[0,0])
        ax_mask = fig.add_subplot(gspec[0,1])

        return fig, (ax_img, ax_mask)


    def plot_img(self, title = ""): 
        img = self.img
        im = self.ax_img.imshow(img, norm = self.divnorm)
        im.set_cmap('seismic')
        plt.colorbar(im, ax = self.ax_img, orientation="horizontal", pad = 0.05)


    def plot_mask(self, title = ""): 
        mask = self.mask
        im = self.ax_mask.imshow(mask, norm = self.divnorm)
        im.set_cmap('seismic')
        plt.colorbar(im, ax = self.ax_mask, orientation="horizontal", pad = 0.05)


    def plot_center(self, center = (0, 0), angle = 0):
        x, y = center
        marker_obj = mpl.markers.MarkerStyle(marker = "+")
        if not angle == None: marker_obj._transform = marker_obj.get_transform().rotate_deg(angle)
        self.ax_img.plot(x, y, marker = marker_obj, markersize = 18, color = 'black', markeredgewidth = 4)


    def config_colorbar(self, vmin = -1, vcenter = 0, vmax = 1):
        # Plot image...
        self.divnorm = mcolors.TwoSlopeNorm(vcenter = vcenter, vmin = vmin, vmax = vmax)


    def show(self, center = None, angle = None): 
        self.fig, (self.ax_img, self.ax_mask) = self.create_panels()

        self.plot_img()
        if isinstance(center, tuple): self.plot_center(center, angle)
        self.plot_mask()
        plt.show()




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




# Specify the dataset and detector...
exp, run, mode, detector_name = 'amo06516', '90', 'idx', 'pnccdFront'

# Initialize an image reader...
img_reader = PsanaImg(exp, run, mode, detector_name)

# Access an image (e.g. event 796)...
event_num = 796
img = img_reader.get(event_num, mode = "image")

## # Normalize image...
## img = (img - np.mean(img)) / np.std(img)

angle = 23
center = (524, 506)
trans_random_rotate = transform.RandomRotate(angle = angle, center = center)
img = trans_random_rotate(img)

# Apply the transform...
num_patch = 5
size_patch_y, size_patch_x = 70, 500
trans_random_patch = transform.RandomPatch(num_patch, size_patch_y,      size_patch_x, 
                                         var_patch_y = 0.2, var_patch_x = 0.2, 
                                         is_return_mask = True, is_random_flip = True)
img_masked, mask = trans_random_patch(img)

## angle = 23
## center = (524, 506)
## trans_random_rotate = transform.RandomRotate(angle = angle, center = center)
## img_masked = trans_random_rotate(img_masked)
## mask = trans_random_rotate(mask)

# Normalize image...
img_masked = (img_masked - np.mean(img_masked)) / np.std(img_masked)

# Dispaly an image...
disp_manager = DisplaySPIImg(img_masked, mask, figsize = (18, 8))
disp_manager.show(center = center, angle = angle)
