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

    def __init__(self, pair_dict, threshold, figsize, **kwargs):
        self.pair_dict = pair_dict
        self.figsize = figsize
        self.threshold = threshold
        for k, v in kwargs.items(): setattr(self, k, v)

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
        self.ax_dict[k].set_ylim(-0.1, 0.4)
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
        self.ax_dict[k].set_ylim(-0.1, 0.4)
        ## self.ax_l.set_xticks([])
        self.ax_dict[k].set_title(' vs '.join([ self.title_dict[i] for i in k ]))
        ## self.ax_dict[k].set_ylabel('Distance')


    def show(self, filename = None): 
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
                    ## bbox_transform=self.fig.transFigure,
                )

            if i == 0 or i == 5: self.ax_dict[k].set_ylabel('Distance')

        ## self.fig.subplots_adjust(wspace=1)

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
            ## plt.savefig(path_pdf, dpi = 100, bbox_inches='tight', pad_inches = 0)
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




def judger(dist, threshold):
    return True if dist < threshold else False




# File to analyze...
## id_log = "20220203150247"    # Scenario 1
## id_log = '20220203115233'    # Scenario 2
## id_log = '20220207115721'    # Convnet
## id_log = '20220214122748'    # Convnet
## id_log = '20220214122748'
## id_log = "20220214220130"
id_log = "20220215220550"

# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
fl_alog = f"{id_log}.validate.pair.alog"
DRCALOG = "analysis"
prefixpath_alog = os.path.join(drc_cwd, DRCALOG)
if not os.path.exists(prefixpath_alog): os.makedirs(prefixpath_alog)
path_alog = os.path.join(prefixpath_alog, fl_alog)

logging.basicConfig( filename = path_alog,
                     filemode = 'w',
                     format="%(message)s",
                     level=logging.INFO, )
logger = logging.getLogger(__name__)

# Locate the path of the log file...
fl_log   = f'{id_log}.validate.pair.log'
drc_log  = 'logs'
path_log = os.path.join(drc_log, fl_log)

# Read the file...
log_dict = read_log(path_log)

# Understand dataset...
res_dict = { (True , True ) : [],    # True  positive
             (False, True ) : [],    # False positive
             (False, False) : [],    # False negative
             (True , False) : [],    # True  negative
           }
resfull_dict = { (True , True ) : [],    # True  positive
                 (False, True ) : [],    # False positive
                 (False, False) : [],    # False negative
                 (True , False) : [],    # True  negative
               }
pair_dict = {}
threshold = 0.25/2

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

    # Assign category...
    res_dict[(predicate, res_pred)].append((i, dist))
    resfull_dict[(predicate, res_pred)].append((label_anchor_supposed, label_second_supposed, dist))

    k = tuple(sorted([label_anchor_supposed, label_second_supposed]))
    if not k in pair_dict: pair_dict[k] = [(i, dist)]
    else: pair_dict[k].append((i, dist))

figsize = (28, 10)
disp_manager = HistValidation(pair_dict = pair_dict, threshold = threshold, figsize = figsize)
fl_pdf = f"alog.{id_log}.brkdwn"
## disp_manager.show()
disp_manager.show(filename = fl_pdf)
