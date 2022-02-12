#!/usr/bin/env python
# -*- coding: utf-8 -*-

import psana
import matplotlib.pyplot       as plt
import matplotlib.colors       as mcolors
import matplotlib.patches      as mpatches
import matplotlib.transforms   as mtransforms
import matplotlib.font_manager as font_manager
import numpy as np
import os


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




class HistSPIImg:

    def __init__(self, img, figsize, **kwargs):
        self.img = img
        self.figsize = figsize
        for k, v in kwargs.items(): setattr(self, k, v)

        self.img_mean = np.nanmean(img)
        self.img_std  = np.std(img)

        self.fig, (self.ax_img, self.ax_norm) = self.create_panels()


    def create_panels(self):
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

        nrows, ncols = 3, 2
        fig = plt.figure(figsize = self.figsize)

        gspec   = fig.add_gridspec(nrows, ncols)
        ax_img  = (fig.add_subplot(gspec[0,0]), fig.add_subplot(gspec[1:,0]))
        ax_norm = (fig.add_subplot(gspec[0,1]), fig.add_subplot(gspec[1:,1]))
        return fig, (ax_img, ax_norm)


    def plot_img(self, rng = [], bin_cap = 10000, vcenter = 0, vmin = 0, vmax = 1): 
        img = self.img
        bin_val, bin_rng = self.population_density(img, bin_cap)

        img_mean = self.img_mean
        img_std  = self.img_std

        if len(rng) == 0: rng = bin_rng[0], bin_rng[-1]
        data = []
        for i in range(len(bin_val)):
            data.append([bin_rng[i],   bin_val[i]])
            data.append([bin_rng[i+1], bin_val[i]])
        data = np.array(data)

        # Plot population density...
        self.ax_img[0].plot(data[:, 0], data[:, 1])
        self.ax_img[0].set_box_aspect(0.5)
        self.ax_img[0].set_yscale('log')
        self.ax_img[0].set_xlim(rng[0], rng[1])
        ymin, ymax = np.nanmin(bin_val), np.nanmax(bin_val)
        self.ax_img[0].set_ylim(ymin, ymax)
        self.ax_img[0].set_title(f"$\mu = {img_mean:.4f}$; $\sigma = {img_std:.4f};  b = {bin_cap}$", fontdict = {"fontsize" : 18})
        self.ax_img[0].set_xlabel('Pixel intensity')
        self.ax_img[0].set_ylabel('Population density')
        print(f"Image: ")
        print(f"Range X: [{bin_rng[0]}, {bin_rng[-1]}]")
        print(f"Range Y: [{bin_val[0]}, {bin_val[-1]}]")

        # Add a rectangle to specify the range (vmin, vmax)...
        mtrans = mtransforms.blended_transform_factory(self.ax_img[0].transData, self.ax_img[0].transAxes)
        #                                              ~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~
        # Axes X (Keep as DC) ______________________________________|                       :
        #                                                                                   :
        # Axes Y (Transform to AC) .........................................................:

        w = vmax - vmin
        rect = mpatches.Rectangle((vmin, 0), width=w, height=1, transform=mtrans,
                                  color='yellow', alpha=0.5)
        if not vcenter == None: self.ax_img[0].add_patch(rect)

        # Plot image...
        divnorm = mcolors.TwoSlopeNorm(vcenter = vcenter, vmin = vmin, vmax = vmax)
        im = self.ax_img[1].imshow(img, norm = divnorm)
        im.set_cmap('seismic')
        plt.colorbar(im, ax = self.ax_img[1],orientation="horizontal", pad = 0.05)


    def plot_norm(self, rng = [], bin_cap = 10000, vcenter = 0, vmin = 0, vmax = 1): 
        img = self.img
        img_mean = self.img_mean
        img_std  = self.img_std
        img_norm = (img - img_mean) / img_std

        img_mean = np.nanmean(img_norm)
        img_std  = np.std(img_norm)
        bin_val, bin_rng = self.population_density(img_norm, bin_cap)

        if len(rng) == 0: rng = bin_rng[0], bin_rng[-1]
        data = []
        for i in range(len(bin_val)):
            data.append([bin_rng[i],   bin_val[i]])
            data.append([bin_rng[i+1], bin_val[i]])
        data = np.array(data)

        # Plot population density...
        self.ax_norm[0].plot(data[:, 0], data[:, 1])
        self.ax_norm[0].set_box_aspect(0.5)
        self.ax_norm[0].set_yscale('log')
        self.ax_norm[0].set_xlim(rng[0], rng[1])
        ymin, ymax = np.nanmin(bin_val), np.nanmax(bin_val)
        self.ax_norm[0].set_ylim(ymin, ymax)
        self.ax_norm[0].set_title(f"$\mu = {img_mean:.4f}$; $\sigma = {img_std:.4f};  b = {bin_cap}$", fontdict = {"fontsize" : 18})
        self.ax_norm[0].set_xlabel('Pixel intensity')
        self.ax_norm[0].set_ylabel('Population density')
        print(f"Norm: ")
        print(f"Range X: [{bin_rng[0]}, {bin_rng[-1]}]")
        print(f"Range Y: [{bin_val[0]}, {bin_val[-1]}]")

        # Add a rectangle to specify the range (vmin, vmax)...
        mtrans = mtransforms.blended_transform_factory(self.ax_norm[0].transData, self.ax_norm[0].transAxes)

        # Add a rectangle to specify the range (vmin, vmax)...
        w = vmax - vmin
        rect = mpatches.Rectangle((vmin, 0), width=w, height=1, transform=mtrans,
                                  color='yellow', alpha=0.5)
        if not vcenter == None: self.ax_norm[0].add_patch(rect)

        # Plot image...
        divnorm = mcolors.TwoSlopeNorm(vcenter = vcenter, vmin = vmin, vmax = vmax)
        im = self.ax_norm[1].imshow(img_norm, norm = divnorm)
        im.set_cmap('seismic')
        plt.colorbar(im, ax = self.ax_norm[1], orientation="horizontal", pad = 0.05)

    def show(self, filename = None): 
        self.plot_img (rng = [-150, 250], bin_cap = 200, vcenter = 0, vmin = -120,  vmax = 150)
        self.plot_norm(rng = [-1,     5], bin_cap = 200, vcenter = 0, vmin = -1,    vmax = 4)

        img_mean = self.img_mean
        img_std  = self.img_std

        ## plt.subplots_adjust(top    = 1.0)
        ## plt.subplots_adjust(bottom = 0.0)
        plt.subplots_adjust(hspace = 0.0)
        if not isinstance(filename, str): 
            plt.show()
        else:
            plt.suptitle(f"IMG - {filename}", y = 0.92)

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
                den = np.nan    # Follow behaviros of matlab and gnuplot that ignore infinity
                                # https://discourse.matplotlib.org/t/plotting-array-with-inf/9641/2
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
## event_num = 796
## event_num = 1997
## event_num = 3120
## event_num = 1
## event_num = 709

event_nums = (1, 709, 796, 1997, 3120)
for event_num in event_nums:
    img = img_reader.get(event_num, mode = "image")
    ## img_raw = img_reader.get(event_num, mode = "raw")

    # Dispaly an image...
    ## img[img < np.nanstd(img)] = 0
    basename = f"{exp}.{int(run):04d}"
    filename = f"{basename}.{event_num:05d}"
    disp_manager = HistSPIImg(img, figsize = (18, 16))
    ## disp_manager.show()
    disp_manager.show(filename = filename)
