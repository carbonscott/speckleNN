#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib              as mpl
import matplotlib.pyplot       as plt
import matplotlib.colors       as mcolors
import matplotlib.patches      as mpatches
import matplotlib.transforms   as mtransforms
import matplotlib.font_manager as font_manager
import numpy as np
import os
from deepprojection.datasets.simulated_square_detector import SPIImgDataset, ConfigDataset
from simulated_square_detector_preprocess              import DatasetPreprocess

class DisplaySPIImg:

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
        nrows, ncols = 2, 2
        fig = plt.figure(figsize = self.figsize)
        gspec =  fig.add_gridspec(nrows, ncols, height_ratios = [1, 1/20])
        ax_img  = fig.add_subplot(gspec[0,0], aspect = 1)
        ax_mask = fig.add_subplot(gspec[0,1], aspect = 1)
        ax_bar_img  = fig.add_subplot(gspec[1,0], aspect = 1/20)
        ax_bar_mask = fig.add_subplot(gspec[1,1], aspect = 1/20)

        return fig, (ax_img, ax_mask, ax_bar_img, ax_bar_mask)


    def plot_img(self): 
        img = self.img
        im = self.ax_img.imshow(img, norm = self.divnorm)
        im.set_cmap('seismic')
        plt.colorbar(im, cax = self.ax_bar_img, orientation="horizontal", pad = 0.05)


    def plot_mask(self): 
        mask = self.mask
        im = self.ax_mask.imshow(mask, norm = self.divnorm)
        im.set_cmap('seismic')
        plt.colorbar(im, cax = self.ax_bar_mask, orientation="horizontal", pad = 0.05)


    def plot_center(self, center = (0, 0), angle = 0):
        x, y = center
        marker_obj = mpl.markers.MarkerStyle(marker = "+")
        if not angle == None: marker_obj._transform = marker_obj.get_transform().rotate_deg(angle)
        ## self.ax_img.plot(x, y, marker = marker_obj, markersize = 18, color = 'black', markeredgewidth = 4)
        self.ax_mask.plot(x, y, marker = marker_obj, markersize = 18, color = 'black', markeredgewidth = 4)


    def config_colorbar(self, vmin = -1, vcenter = 0, vmax = 1):
        # Plot image...
        self.divnorm = mcolors.TwoSlopeNorm(vcenter = vcenter, vmin = vmin, vmax = vmax)


    def show(self, center = None, angle = None, title = '', is_save = False): 
        self.fig, (self.ax_img, self.ax_mask, self.ax_bar_img, self.ax_bar_mask) = self.create_panels()

        self.plot_img()
        if isinstance(center, tuple): self.plot_center(center, angle)
        self.plot_mask()

        ## if isinstance(title, str) and len(title) > 0: plt.suptitle(title)

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
            ## plt.savefig(path_pdf, dpi = 100, bbox_inches='tight', pad_inches = 0)
            plt.savefig(path_pdf, dpi = 100, transparent=True)

        plt.close('all')




# Config the dataset...
exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ConfigDataset.NOHIT, ConfigDataset.BACKGROUND ]
config_dataset = ConfigDataset( fl_csv            = 'simulated.square_detector.datasets.pdb_not_sampled.common.csv',
                                size_sample       = 100,
                                seed              = 0,
                                trans             = None,
                                frac_train        = 0.2,
                                dataset_usage     = 'train',
                                exclude_labels    = exclude_labels, )

# Create image manager...
spiimg = SPIImgDataset(config_dataset)
img, _ = spiimg.get_img_and_label(0)

# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
dataset_preproc = DatasetPreprocess(config_dataset, sigma = 0.15 * 100)
trans = dataset_preproc.config_trans()

# Read an image...
for idx in range(0,30):
    fl_base, id_frame, label = spiimg.imglabel_list[idx]

    # Don't apply those changes from configuration...
    spiimg.trans = None
    img, _ = spiimg[idx]
    img = img.squeeze(axis = 0)

    # Apply those changes from configuration...
    spiimg.trans = trans

    # Get transed image...
    img_masked, _ = spiimg[idx]
    img_masked = img_masked.squeeze(axis = 0)

    # None rotation...
    center = None
    angle  = None

    title = f'simulated.{fl_base}.{id_frame}.{label}'

    # Dispaly an image...
    title = f'simulated.sq.{idx:06d}'
    disp_manager = DisplaySPIImg(img, img_masked, figsize = (18, 8))
    disp_manager.show(center = center, angle = angle, title = title, is_save = False)
    ## disp_manager.show(center = center, angle = angle, title = title, is_save = True)
