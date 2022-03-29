#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
from sklearn.manifold import TSNE
from deepprojection.utils import read_log, set_seed
import matplotlib              as mpl
import matplotlib.pyplot       as plt
import matplotlib.colors       as mcolors
import matplotlib.patches      as mpatches
import matplotlib.transforms   as mtransforms
import matplotlib.font_manager as font_manager

set_seed(0)

class VizEmb:

    def __init__(self, emb, des, title, figsize, **kwargs):
        self.emb     = emb
        self.des     = des
        self.title   = title
        self.figsize = figsize
        for k, v in kwargs.items(): setattr(self, k, v)

        self.config_fonts()

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
        nrows, ncols = 1, 1
        fig = plt.figure(figsize = self.figsize)

        gspec   = fig.add_gridspec(nrows, ncols)
        ax_list = (fig.add_subplot(gspec[0,0]), )
        return fig, ax_list


    def plot_by_label(self): 
        ax = self.ax_list[0]
        des_list_flat = [ des[0] for des in des_list ]

        # Find ranges...
        vmin, vmax = np.min(self.emb), np.max(self.emb)

        c_list = ( ("1", "red"      , "single hit"),
                   ("2", "#003f5c"  , "double hit"), 
                   ("3", "#bc5090"  , "triplet hit"), 
                   ("4", "#ffa600"  , "quadruple hit"), )

        for c, fc, l in c_list:
            idx_list = [ des[des.rfind('_hit')-1] == c for des in des_list_flat ]
            x = self.emb[idx_list,0]
            y = self.emb[idx_list,1]
            ax.scatter(x, y, facecolor = fc, alpha = 0.6, label = l)

        ax.set_xlim(1.10 * vmin, 1.10 * vmax)
        ax.set_ylim(1.10 * vmin, 1.10 * vmax)
        ax.set_aspect(1)
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(
                   edgecolor = "none",
                   ncol      = 1,
                   loc       = "best",
                   ## bbox_to_anchor=(0.0, 1.5)
        )


    def show(self, filename = None): 
        self.fig, self.ax_list = self.create_panels()

        self.plot_by_label()

        plt.suptitle(self.title, y = 0.95)
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




# File to analyze...
## timestamp = "20220326123119"
## timestamp = "20220326123157"
timestamp = "20220324100158"

# Validate mode...
istrain = False
mode_validate = 'train' if istrain else 'test'

# [[[ LOAD LOG ]]]

# Locate the path of the log file...
basename = f'{timestamp}.validate.simple.{mode_validate}'
fl_log   = f'{basename}.log'
drc_log  = 'logs'
path_log = os.path.join(drc_log, fl_log)

# Read the file...
log_dict = read_log(path_log)
des_list = log_dict['data']


# [[[ LOAD EMBEDDING ]]]
fl_emb = f"{basename}.pt"
DRCEMB = "embeds"
drc_cwd = os.getcwd()
prefixpath_emb = os.path.join(drc_cwd, DRCEMB)
if not os.path.exists(prefixpath_emb): os.makedirs(prefixpath_emb)
path_emb = os.path.join(prefixpath_emb, fl_emb)

imgs = torch.load(path_emb)
imgs_np = imgs.squeeze(1).numpy()

# [[[ CLUSTERING ]]]
imgs_latent = TSNE( n_components = 2, init = "random" ).fit_transform(imgs_np)

filename = f"cluster.{timestamp}"
disp_manager = VizEmb(emb = imgs_latent, des = des_list, title = '', figsize = (12, 10))
disp_manager.show()
disp_manager.show(filename = filename)
