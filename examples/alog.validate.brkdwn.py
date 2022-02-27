#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging
import os
from deepprojection.utils import read_log
import matplotlib.pyplot       as plt
import matplotlib.colors       as mcolors
import matplotlib.patches      as mpatches
import matplotlib.transforms   as mtransforms
import matplotlib.font_manager as font_manager


class HistValidation:

    def __init__(self, timestamp, pair_dict, istrain, threshold, figsize, **kwargs):
        self.timestamp = timestamp
        self.pair_dict = pair_dict
        self.istrain = istrain
        self.figsize = figsize
        self.threshold = threshold
        for k, v in kwargs.items(): setattr(self, k, v)

        self.mode_validate = 'train' if istrain else 'test'

        self.config_fonts()

        self.title_dict = {
            '1' : 'single',
            '2' : 'multi',
            '0' : 'not sample',
            '9' : 'background',
        }

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
        nrows, ncols = 2, 5
        fig = plt.figure(figsize = self.figsize)

        gspec = fig.add_gridspec(nrows, ncols)

        k = tuple(self.pair_dict.keys())
        ax_dict = {}
        for row in range(nrows):
            for col in range(ncols):
                i = row * ncols + col
                k_i = k[i]
                ax_dict[k_i] = fig.add_subplot(gspec[row,col])

        return fig, ax_dict


    def plot_dist_pos(self, k, marker = "o", edgecolor = "none", facecolor = '#263238', title = ""): 
        x = [ line[0] for line in self.pair_dict[k] if line[1] < self.threshold ]
        y = [ line[1] for line in self.pair_dict[k] if line[1] < self.threshold ]
        self.ax_dict[k].scatter(x, y, marker = marker, 
                                edgecolor = edgecolor, 
                                facecolor = facecolor, 
                                alpha = 1.00, 
                                label = 'Positive (predicted by model)',
                                s = 72 * 2.5, )
        self.ax_dict[k].set_box_aspect(0.5)
        self.ax_dict[k].set_xlabel('Validation Example Index')
        self.ax_dict[k].set_ylim(-0.1, 4.1)
        ## self.ax_l.set_xticks([])
        self.ax_dict[k].set_title(' vs '.join([ self.title_dict[i] for i in k ]))
        ## self.ax_dict[k].set_ylabel('Distance')


    def plot_dist_neg(self, k, marker = "o", edgecolor = "none", facecolor = '#E53935', title = ""): 
        x = [ line[0] for line in self.pair_dict[k] if line[1] >= self.threshold ]
        y = [ line[1] for line in self.pair_dict[k] if line[1] >= self.threshold ]
        self.ax_dict[k].scatter(x, y, marker = marker, 
                                edgecolor = edgecolor, 
                                facecolor = facecolor, 
                                alpha = 1.00, 
                                label = 'Negative (predicted by model)',
                                s = 72 * 2.5, )
        self.ax_dict[k].set_box_aspect(0.5)
        self.ax_dict[k].set_xlabel('Validation Example Index')
        self.ax_dict[k].set_ylim(-0.1, 4.1)
        ## self.ax_l.set_xticks([])
        self.ax_dict[k].set_title(' vs '.join([ self.title_dict[i] for i in k ]))
        ## self.ax_dict[k].set_ylabel('Distance')


    def show(self, is_save = False): 
        self.fig, self.ax_dict = self.create_panels()

        for i, k in enumerate(self.pair_dict.keys()):
            self.plot_dist_neg(k, marker = "o", facecolor = 'none', edgecolor = '#E53935')
            self.plot_dist_pos(k, marker = "o", facecolor = 'none', edgecolor = '#43A047')

            if i == 0: 
                self.ax_dict[k].legend(
                    edgecolor="none",
                    ncol=1,
                    loc="upper left",
                    bbox_to_anchor=(0.0, 1.8),
                    borderaxespad=0,
                    framealpha=0.0,
                    ## bbox_transform=self.fig.transFigure,
                )

            if i == 0 or i == 5: self.ax_dict[k].set_ylabel('Distance')

        ## self.fig.subplots_adjust(wspace=1)

        if not is_save: 
            plt.show()
        else:
            # Set up drc...
            DRCPDF         = "pdfs"
            drc_cwd        = os.getcwd()
            prefixpath_pdf = os.path.join(drc_cwd, DRCPDF)
            if not os.path.exists(prefixpath_pdf): os.makedirs(prefixpath_pdf)

            # Specify file...
            fl_pdf = f"alog.{self.timestamp}.{self.mode_validate}.brkdwn.pdf"
            path_pdf = os.path.join(prefixpath_pdf, fl_pdf)

            # Export...
            ## plt.savefig(path_pdf, dpi = 100, bbox_inches='tight', pad_inches = 0)
            plt.savefig(path_pdf, dpi = 100, transparent=True)


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




def judger(dist, threshold):
    return True if dist < threshold else False




# File to analyze...
## timestamp = "20220225214556"    # size_sample : 500, alpha : 2.6
## timestamp = "20220225214804"    # 2000, 2.6
## timestamp = "20220225214825"    # 2000, 2.0
## timestamp = "20220225235026"    # 1500, 2.6
## timestamp = "20220225234941"    # 1500, 2.0
## timestamp = "20220225235155"    # 1500, 2.2
## timestamp = "20220225235242"    # 1500, 1.6
## timestamp = "20220225235121"    # 1500, 2.4
## timestamp = "20220225234917"    # 2000, 1.6
## timestamp = "20220225234902"    # 2000, 2.4
timestamp = "20220225234851"    # 2000, 2.2
## timestamp = "20220225234840"    # 2000, 1.8

# Validate mode...
istrain = False
mode_validate = 'train' if istrain else 'test'

# Locate the path of the log file...
fl_log   = f'{timestamp}.validate.{mode_validate}.log'
drc_log  = 'logs'
path_log = os.path.join(drc_log, fl_log)

# Read the file...
log_dict = read_log(path_log)

# Understand dataset...
pair_dict = {}
threshold = 1.0

for i, (anchor, second, dist) in enumerate(log_dict[ "data" ]):
    anchor = tuple(anchor.split())
    second = tuple(second.split())
    dist   = np.float64(dist)

    # Get the supposed result (match or not)...
    label_anchor_supposed = anchor[-1]
    label_second_supposed = second[-1]
    res_supposed = label_anchor_supposed == label_second_supposed

    # Get the predicted result (match or not)...
    res_pred = judger(dist, threshold)

    # Get the predicate...
    predicate = res_supposed == res_pred

    k = tuple(sorted([label_anchor_supposed, label_second_supposed]))
    if not k in pair_dict: pair_dict[k] = [(i, dist)]
    else: pair_dict[k].append((i, dist))

# Sort dictionary by keys...
pair_dict = {k: v for k, v in sorted(pair_dict.items(), key=lambda item: item[0])}

figsize = (28, 10)
disp_manager = HistValidation(timestamp = timestamp, pair_dict = pair_dict, istrain = istrain, threshold = threshold, figsize = figsize)
disp_manager.show()
disp_manager.show(is_save = True)
