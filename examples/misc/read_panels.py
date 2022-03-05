#!/usr/bin/env python
# -*- coding: utf-8 -*-

import psana
import matplotlib.pyplot as plt
import numpy as np
import os
from deepprojection.datasets.experiments import PsanaImg
from deepprojection.datasets.panels import PsanaPanel
from deepprojection.datasets.transform import hflip

class DisplaySPIImg():

    def __init__(self, img_img, img_raw, figsize, **kwargs):
        self.img_img = img_img
        self.img_raw = img_raw
        self.figsize = figsize
        for k, v in kwargs.items(): setattr(self, k, v)



    def create_panels(self):
        plt.rcParams.update({'font.size': 18})
        plt.rcParams.update({'font.family' : 'sans-serif'})
        ## fig, (ax_img, ax_raw) = plt.subplots(nrows = 2, ncols = 4, figsize = self.figsize)
        nrows, ncols = 2, 5
        fig = plt.figure(figsize = self.figsize)
        gspec =  fig.add_gridspec(nrows, ncols, width_ratios = [1, 1, 1, 1, 1/12])

        ax_img = fig.add_subplot(gspec[0:2,0:2])

        ax_raws = []
        ax_raws.append(fig.add_subplot(gspec[0,2]))
        ax_raws.append(fig.add_subplot(gspec[0,3]))
        ax_raws.append(fig.add_subplot(gspec[1,2]))
        ax_raws.append(fig.add_subplot(gspec[1,3]))

        ax_bar = fig.add_subplot(gspec[:,4])
        return fig, (ax_img, ax_raws, ax_bar)


    def plot_img(self, title = ""): 
        img_img = self.img_img
        img_img = (img_img - np.mean(img_img)) / np.std(img_img)
        im = self.ax_img.imshow(img_img, vmin = -1, vmax = 1)
        im.set_cmap('seismic')
        plt.colorbar(im, cax = self.ax_bar, orientation="vertical", pad = 0.5)


    def plot_raw(self, title = ""): 
        img_raw = self.img_raw
        img_raw = (img_raw - np.mean(img_raw)) / np.std(img_raw)

        for pos, ax_raw in enumerate(self.ax_raws):
            ## img_raw[pos] = (img_raw[pos] - np.mean(img_raw)) / np.std(img_raw)
            im = ax_raw.imshow(img_raw[pos], vmin = -1, vmax = 1)
            im.set_cmap('seismic')


    def show(self, title = '', is_save = False): 
        self.fig, (self.ax_img, self.ax_raws, self.ax_bar) = self.create_panels()

        self.plot_img()
        self.plot_raw()

        if not is_save: 
            plt.show()
        else:
            # Set up drc...
            DRCPDF         = "pdfs"
            drc_cwd        = os.getcwd()
            prefixpath_pdf = os.path.join(drc_cwd, DRCPDF)
            if not os.path.exists(prefixpath_pdf): os.makedirs(prefixpath_pdf)

            # Specify file...
            fl_pdf = f"{title}.pdf"
            path_pdf = os.path.join(prefixpath_pdf, fl_pdf)

            # Export...
            plt.savefig(path_pdf, dpi = 100, transparent=True)




# Specify the dataset and detector...
exp, run, mode, detector_name = 'amo06516', '102', 'idx', 'pnccdFront'

# Initialize an image reader...
img_reader = PsanaImg(exp, run, mode, detector_name)
pan_reader = PsanaPanel(exp, run, mode, detector_name)

# Access an image (e.g. event 796)...
event_num = 6025    # 102
img_img = img_reader.get(event_num, mode = "image")

# Standardizing rule...
trans_stand_dict = { "1" : hflip, "3" : hflip }

# Read all panels and apply transformation to standardize...
num_panels = 4
imgs_calib = []
for id_panel in [ str(i) for i in range(num_panels) ]:
    img_calib = pan_reader.get(event_num, id_panel, mode = "calib")
    if id_panel in trans_stand_dict:
        img_calib = trans_stand_dict[id_panel](img_calib)
    imgs_calib.append(img_calib)

title = f'panel.{exp}.{int(run):04d}.{event_num:06d}'

# Dispaly an image...
disp_manager = DisplaySPIImg(img_img, imgs_calib, figsize = (20, 8))
disp_manager.show(title = title, is_save = False)
disp_manager.show(title = title, is_save = True)
