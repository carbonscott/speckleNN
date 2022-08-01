#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import csv

from image_preprocess import DatasetPreprocess
from deepprojection.plugins import PsanaImg
from math import ceil

import matplotlib              as mpl
import matplotlib.pyplot       as plt
import matplotlib.colors       as mcolors
import matplotlib.patches      as mpatches
import matplotlib.transforms   as mtransforms
import matplotlib.font_manager as font_manager

class VizFalse:

    def __init__(self, data, dist_ary, title, figsize, **kwargs):
        self.data     = data
        self.dist_ary = dist_ary
        self.title    = title
        self.figsize  = figsize
        for k, v in kwargs.items(): setattr(self, k, v)

        self.config_fonts()
        self.config_colorbar()

        self.min_idx = self.find_min_idx_seq()

        return None


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
        ncols = 4
        nrows = ceil(len(self.data) / ncols)

        fig   = plt.figure(figsize = self.figsize)
        gspec = fig.add_gridspec( nrows, ncols,
                                  ## width_ratios  = [1, 4/20, 4/20, 4/20, 4/20, 4/20, 1/20],·
                                  ## height_ratios = [4/20, 4/20, 4/20, 4/20, 4/20], 
                                )

        ## ax_list  = (fig.add_subplot(gspec[0,0:ncols], aspect = 1), )
        ## ax_list += tuple(fig.add_subplot(gspec[i,1 + j], aspect = 1) for i in range(5) for j in range(5))
        ## ax_list += (fig.add_subplot(gspec[-1,0:ncols]), )
        ax_list = [ fig.add_subplot(gspec[i, j]) for i in range(nrows) for j in range(ncols) ]

        self.ncols = ncols
        self.nrows = nrows

        return fig, ax_list


    def config_colorbar(self, vmin = -1, vcenter = 0, vmax = 1):
        # Plot image...
        self.divnorm = mcolors.TwoSlopeNorm(vcenter = vcenter, vmin = vmin, vmax = vmax)


    def plot(self, idx_seq):
        ax_img  = self.ax_list[idx_seq]
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        if idx_seq < len(self.data):
            data = self.data[idx_seq]

            ## ax_cbar = self.ax_list[-1]
            im = ax_img.imshow(data, norm = self.divnorm)
            im.set_cmap('seismic')
            ## plt.colorbar(im, cax = ax_cbar, orientation="vertical", pad = 0.05)

            for axis in ['top', 'bottom', 'left', 'right']:
                ax_img.spines[axis].set_linewidth(4.0)

            if idx_seq in self.min_idx:
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax_img.spines[axis].set_color('green')
            if idx_seq % self.ncols == 0:
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax_img.spines[axis].set_color('red')
            ## if idx_seq % 2 == 1:
            ##     for axis in ['top', 'bottom', 'left', 'right']:
            ##         ax_img.spines[axis].set_color('green')
            ## else:
            ##     for axis in ['top', 'bottom', 'left', 'right']:
            ##         ax_img.spines[axis].set_color('red')

        else:
            ax_img.set_frame_on(False)

        return None


    def put_text(self, idx_seq):
        dist_sup = self.dist_ary[idx_seq]
        ax_img = self.ax_list[idx_seq]
        ax_img.text(
            0.03,
            0.05,
            f"{dist_sup:6.4f}",
            color="white",
            transform = ax_img.transAxes,
            bbox={"facecolor": "black", "edgecolor": "None", "pad": 1, "alpha": 0.75},
        )


    def find_min_idx_seq(self):
        dist_ary = self.dist_ary
        dist_ary_by_line = dist_ary.reshape(-1,4)
        min_idx_by_line = np.nanargmin(dist_ary_by_line, axis = -1)

        num_lines = len(dist_ary_by_line)
        offset_idx_list = [ i * 4 for i in range(num_lines) ]

        min_idx_ary = offset_idx_list + min_idx_by_line

        return min_idx_ary


    def adjust_margin(self):
        self.fig.subplots_adjust(
            top=1-0.049,
            bottom=0.049,
            left=0.042,
            right=1-0.042,
            hspace=0.2,
            wspace=0.2
        )


    def show(self, filename = None):
        self.fig, self.ax_list = self.create_panels()

        num_plot = self.ncols * self.nrows

        for idx_seq in range(num_plot):
            self.plot(idx_seq)

            if idx_seq % self.ncols:
                self.put_text(idx_seq)

        ## plt.tight_layout()
        self.adjust_margin()

        ## plt.suptitle(self.title, y = 0.95)
        if not isinstance(filename, str): 
            plt.show()
        else:
            # Set up drc...
            DRCPDF         = "pdfs"
            drc_cwd        = os.getcwd()
            prefixpath_pdf = os.path.join(drc_cwd, DRCPDF)
            if not os.path.exists(prefixpath_pdf): os.makedirs(prefixpath_pdf)

            # Specify file...
            fl_pdf = f"{filename}.pdf"
            path_pdf = os.path.join(prefixpath_pdf, fl_pdf)

            # Export...
            plt.savefig(path_pdf, dpi = 100)



## # Create fake data
## np.random.seed(0)
## fake_data = np.random.rand(9, 2, 32, 32).reshape(-1, 32, 32)
## 
## disp_manager = VizFalse(fake_data, title = 'fake', figsize = (12,21))
## disp_manager.show()
## 
## stop

# Load images from csv file...
path_csv            = 'inference.csv'
psana_reader_dict = {}
mode              = 'idx'
detector_name     = 'pnccdFront'

with open(path_csv, 'r') as fh:
    lines = csv.reader(fh)
    num_lines = (sum(1 for line in lines) - 1)

with open(path_csv, 'r') as fh:
    lines = csv.reader(fh)
    next(lines)

    for idx_line, line in enumerate(lines):
        tag_query = line[0]

        # Load query image...
        exp, run, event_idx, _ = tag_query.split()

        basename = exp, run

        if not basename in psana_reader_dict:
            psana_reader_dict[basename] = PsanaImg(exp, run, mode, detector_name)

        img_query = psana_reader_dict[basename].get(event_idx)

        # Initialize the data array after the first pair is processed...
        if idx_line == 0:
            # Data preprocessing can be lengthy and defined in dataset_preprocess.py
            img_orig        = img_query
            dataset_preproc = DatasetPreprocess(img_orig)
            trans           = dataset_preproc.config_trans()
            img_trans       = trans(img_orig)
            size_y, size_x  = img_trans.shape
            img_ary         = np.zeros((num_lines * len(line), size_y, size_x))
            dist_ary        = np.zeros((num_lines * len(line)))
            dist_ary[:]     = np.nan    # To facilitate finding min dist

        # Track query image...
        img_query = trans(img_query)
        img_ary[idx_line * len(line)] = (img_query - img_query.mean()) / img_query.std()

        # Load support image...
        for idx_sup, dat_sup in enumerate(line[1:]):
            tag_sup, dist_sup = dat_sup.split(':')

            exp, run, event_idx, _ = tag_sup.split()

            basename = exp, run

            if not basename in psana_reader_dict:
                psana_reader_dict[basename] = PsanaImg(exp, run, mode, detector_name)

            img_sup = psana_reader_dict[basename].get(event_idx)
            img_sup = trans(img_sup)

            # Track sup image...
            img_ary[idx_line * len(line) + idx_sup + 1] = (img_sup - img_sup.mean()) / img_sup.std()

            # Track the dist...
            dist_ary[idx_line * len(line) + idx_sup + 1] = float(dist_sup)

fl_pdf  = f'inference'
disp_manager = VizFalse(img_ary, dist_ary, title = '', figsize = (8.5, 5.7))
disp_manager.show()
disp_manager.show(filename = fl_pdf)
