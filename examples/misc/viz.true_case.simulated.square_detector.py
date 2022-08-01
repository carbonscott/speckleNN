#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import csv
import h5py
import random

from simulated_square_detector_preprocess import DatasetPreprocess
from math import ceil

import matplotlib              as mpl
import matplotlib.pyplot       as plt
import matplotlib.colors       as mcolors
import matplotlib.patches      as mpatches
import matplotlib.transforms   as mtransforms
import matplotlib.font_manager as font_manager

class VizFalse:

    def __init__(self, data, title, figsize, **kwargs):
        self.data    = data
        self.title   = title
        self.figsize = figsize
        for k, v in kwargs.items(): setattr(self, k, v)

        self.config_fonts()
        self.config_colorbar()

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
        ncols = 6
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
                ax_img.spines[axis].set_linewidth(2.0)

            if idx_seq % 2 == 1:
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax_img.spines[axis].set_color('green')
            else:
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax_img.spines[axis].set_color('red')

        else:
            ax_img.set_frame_on(False)

        return None


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

random.seed(0)

# Load images from csv file...
timestamp         = '2022_0719_2156_29'
false_label       = '1'
drc_h5            = '/reg/data/ana03/scratch/cwang31/scratch/skopi/h5s_mini.sq'
drc               = 'false_items'
fl_csv            = f'{timestamp}.{false_label}.true.csv'
path_csv          = os.path.join(drc, fl_csv)
psana_reader_dict = {}
mode              = 'idx'
detector_name     = 'pnccdFront'

false_pair_list = []
with open(path_csv, 'r') as fh:
    lines = csv.reader(fh)
    next(lines)

    false_pair_list = [ line for line in lines ]

select_pair_num = 20
false_pair_random_list = random.sample(false_pair_list, select_pair_num)
for idx_line, line in enumerate(false_pair_random_list):
    tag_query, tag_support = line

    # Load query image...
    fl_h5, event_idx, _ = tag_query.split()

    path_h5 = os.path.join(drc_h5, f'{fl_h5}.h5')
    with h5py.File(path_h5, 'r') as fh:
        img_query = fh.get('photons')[int(event_idx)].squeeze(axis = 0)

    # Load support image...
    fl_h5, event_idx, _ = tag_support.split()

    path_h5 = os.path.join(drc_h5, f'{fl_h5}.h5')
    with h5py.File(path_h5, 'r') as fh:
        img_support = fh.get('photons')[int(event_idx)].squeeze(axis = 0)

    # Initialize the data array after the first pair is processed...
    if idx_line == 0:
        # Data preprocessing can be lengthy and defined in dataset_preprocess.py
        img_orig        = img_query
        dataset_preproc = DatasetPreprocess(img_orig)
        trans           = dataset_preproc.config_trans()
        img_trans       = trans(img_orig)
        size_y, size_x  = img_trans.shape
        img_ary         = np.zeros((select_pair_num * 2, size_y, size_x))

    img_query   = trans(img_query)
    img_support = trans(img_support)

    img_ary[idx_line * 2    ] = (img_query   - img_query.mean()  ) / img_query.std()
    img_ary[idx_line * 2 + 1] = (img_support - img_support.mean()) / img_support.std()

# For better visual...
img_ary[img_ary < 0.0] = 0.0

fl_pdf  = f'{timestamp}.{false_label}.true'
disp_manager = VizFalse(img_ary, title = 'false labeled', figsize = (8.5, 10))
disp_manager.show()
disp_manager.show(filename = fl_pdf)
