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
from deepprojection.datasets.mosaic import SPIMosaicDataset, ConfigDataset
from mosaic_preprocess import DatasetPreprocess

class DisplaySPIImg():

    def __init__(self, imgs, img_mosaic, figsize, **kwargs):
        self.imgs       = imgs
        self.img_mosaic = img_mosaic
        self.figsize    = figsize
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
        num_imgs = len(self.imgs)
        nrows, ncols = num_imgs + 1, 2
        fig          = plt.figure(figsize = self.figsize)
        gspec        = fig.add_gridspec(nrows, ncols, width_ratios = [1, 1])
        ax_panels    = [ fig.add_subplot(gspec[i, 0], aspect = 1) for i in range(num_imgs) ]
        ax_bar_img   = fig.add_subplot(gspec[-1,   0], aspect = 1/10)

        ax_mosaic     = fig.add_subplot(gspec[0:num_imgs, 1], aspect = 1)
        ax_bar_mosaic = fig.add_subplot(gspec[-1,   1], aspect = 1/10)

        return fig, (ax_panels, ax_bar_img, ax_mosaic, ax_bar_mosaic)


    def plot_img(self, title = ""): 
        imgs      = self.imgs
        ax_panels = self.ax_panels
        for ax_img, img in zip(ax_panels, imgs):
            im = ax_img.imshow(img, norm = self.divnorm)
            im.set_cmap('seismic')
        cbar = plt.colorbar(im, cax = self.ax_bar_img, orientation="horizontal", pad = 0.05)
        cbar.set_ticks(cbar.get_ticks()[::2])


    def plot_mosaic(self, title = ""): 
        img_mosaic = self.img_mosaic
        im = self.ax_mosaic.imshow(img_mosaic, norm = self.divnorm)
        im.set_cmap('seismic')
        cbar = plt.colorbar(im, cax = self.ax_bar_mosaic, orientation="horizontal", pad = 0.05)
        cbar.set_ticks(cbar.get_ticks()[::2])


    def config_colorbar(self, vmin = -1, vcenter = 0, vmax = 1):
        # Plot image...
        self.divnorm = mcolors.TwoSlopeNorm(vcenter = vcenter, vmin = vmin, vmax = vmax)


    def show(self, center = None, angle = None, title = '', is_save = False): 
        self.fig, (self.ax_panels, self.ax_bar_img, self.ax_mosaic, self.ax_bar_mosaic) = self.create_panels()

        self.plot_img()
        self.plot_mosaic()

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
panels_ordered = [0, 1]
exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP ]
config_dataset = ConfigDataset( fl_csv            = 'datasets.csv',
                                size_sample       = 2000, 
                                psana_mode        = 'calib',
                                resize            = None,
                                seed              = 0,
                                isflat            = False,
                                dataset_usage     = 'train',
                                trans             = None,
                                panels_ordered    = panels_ordered,
                                frac_train        = 0.7,
                                exclude_labels    = exclude_labels, )

# Create image manager...
spiimg = SPIMosaicDataset(config_dataset)
img, _ = spiimg[0]
panel = img[0]

# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
dataset_preproc = DatasetPreprocess(panel, panels_ordered)
trans = dataset_preproc.config_trans()

# Read an image...
for idx in range(10):
    # Don't apply those changes from configuration...
    spiimg.trans     = None
    spiimg.MOSAIC_ON = False
    imgs, _ = spiimg[idx]
    imgs = imgs.squeeze(axis = 0)

    exp, run, event_num, label = spiimg.imglabel_list[idx]

    # Apply those changes from configuration...
    spiimg.trans = trans
    spiimg.MOSAIC_ON = True

    # Get transed image...
    img_mosaic, _ = spiimg[idx]
    img_mosaic = img_mosaic.squeeze(axis = 0)

    # None rotation...
    center = None
    angle  = None

    title = f'mosaic.{exp}.{int(run):04d}.{int(event_num):06d}'

    # Normalize image...
    img_mosaic = (img_mosaic - np.mean(img_mosaic)) / np.std(img_mosaic)

    # Dispaly an image...
    title = f'imagemosaic.{idx:06d}'
    disp_manager = DisplaySPIImg(imgs, img_mosaic, figsize = (8, 10))
    disp_manager.show(center = center, angle = angle, title = title, is_save = False)
    ## disp_manager.show(center = center, angle = angle, title = title, is_save = True)
