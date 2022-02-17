#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot       as plt
import matplotlib.colors       as mcolors
import matplotlib.patches      as mpatches
import matplotlib.transforms   as mtransforms
import matplotlib.font_manager as font_manager


class VizSVD:

    def __init__(self, **kwargs):
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


    def plot_s(self, s, figsize):
        # Set up canvas...
        nrows, ncols = 1, 1
        fig     = plt.figure(figsize = figsize)
        gspec   = fig.add_gridspec(nrows, ncols)
        ax_img  = fig.add_subplot(gspec[0,0])

        # Plot singular values...
        ax_img.plot(s)
        ax_img.set_yscale('log')
        plt.show()


    def plot_u(self, u, dim_img, figsize, vcenter, vmin, vmax):
        # Set up canvas...
        nrows, ncols = 1, 1
        fig     = plt.figure(figsize = figsize)
        gspec   = fig.add_gridspec(nrows, ncols)
        ax_img  = fig.add_subplot(gspec[0,0])

        # Reshape u...
        img = u.reshape(dim_img)

        # Plot image...
        divnorm = mcolors.TwoSlopeNorm(vcenter = vcenter, vmin = vmin, vmax = vmax)
        im = ax_img.imshow(img, norm = divnorm)
        im.set_cmap('seismic')
        plt.colorbar(im, ax = ax_img,orientation="horizontal", pad = 0.05)
        plt.show()





# Read SVD npy...
drc_npy        = "npys"
drc_cwd        = os.getcwd()
prefixpath_npy = os.path.join(drc_cwd, drc_npy)
path_u  = os.path.join(prefixpath_npy, "u.npy")
path_s  = os.path.join(prefixpath_npy, "s.npy")
path_vh = os.path.join(prefixpath_npy, "vh.npy")
u  = np.load(path_u)
s  = np.load(path_s)
vh = np.load(path_vh)

# Read dimension npy...
path_size_y = os.path.join(prefixpath_npy, "size_y.npy")
path_size_x = os.path.join(prefixpath_npy, "size_x.npy")
size_y = np.load(path_size_y)
size_x = np.load(path_size_x)
dim_img = size_y, size_x

# Read the metadata
path_imglabel_list = os.path.join(prefixpath_npy, "imglabel_list.npy")
imglabel_list = np.load(path_imglabel_list)
imglabel_list = tuple(map(tuple, imglabel_list))    # Make it tuple of tuples

vizsvd = VizSVD()
vizsvd.plot_s(s, figsize = (12, 12))
for i in range(20):
    vizsvd.plot_u(u[:, i], figsize = (12, 12), dim_img = dim_img, vcenter = 0, vmin = -1e-2, vmax = 1e-2)
