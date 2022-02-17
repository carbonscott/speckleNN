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


class VizRaw:

    def __init__(self, res_dict, figsize, **kwargs):
        self.res_dict = res_dict
        self.figsize  = figsize
        for k, v in kwargs.items(): setattr(self, k, v)

        count_dict = {}
        for i in (True, False):
            for j in (True, False):
                k = i, j
                count_dict[k] = len(res_dict[k])
        self.count_dict = count_dict

        # Get the total item...
        self.size_samples = sum( count_dict.values() )

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
        nrows, ncols = 1, 2
        fig = plt.figure(figsize = self.figsize)
        gspec   = fig.add_gridspec(nrows, ncols)
        ax_l  = fig.add_subplot(gspec[0,0])
        ax_r = fig.add_subplot(gspec[0,1])
        return fig, (ax_l, ax_r)


    def plot_l(self, cat = (True, True), marker = "o", edgecolor = "none", facecolor = '#263238', label = 'True positive', title = ""): 
        x = [ line[0] for line in self.res_dict[cat] ]
        y = [ line[1] for line in self.res_dict[cat] ]
        self.ax_l.scatter(x, y, marker = marker, 
                                edgecolor = edgecolor, 
                                facecolor = facecolor, 
                                alpha = 1.00, 
                                label = label,
                                s = 72 * 2.5, )
        self.ax_l.set_box_aspect(0.5)
        self.ax_l.set_xlabel('Validation Example Index')
        self.ax_l.set_ylim(-0.1, 0.4)
        ## self.ax_l.set_xticks([])
        self.ax_l.set_ylabel('Distance')


    def plot_r(self, cat = (True, True), marker = "o", edgecolor = "none", facecolor = '#263238', label = 'True positive', title = ""): 
        x = [ line[0] for line in self.res_dict[cat] ]
        y = [ line[1] for line in self.res_dict[cat] ]
        self.ax_r.scatter(x, y, marker = marker, 
                                edgecolor = edgecolor, 
                                facecolor = facecolor, 
                                alpha = 1.00, 
                                label = label,
                                s = 72 * 2, )
        self.ax_r.set_box_aspect(0.5)
        self.ax_r.set_xlabel('Validation Example Index')
        self.ax_r.set_ylim(-0.1, 0.4)
        ## self.ax_r.set_xticks([])
        self.ax_r.set_ylabel('Distance')


    def plot_legend(self):
        self.ax_l.legend(
            edgecolor="none",
            ncol=1,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.4),
            borderaxespad=0,
        )
        self.ax_r.legend(
            edgecolor="none",
            ncol=1,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.4),
            borderaxespad=0,
        )


    def report_metrics(self):
        tp = self.count_dict[(True , True)]
        tn = self.count_dict[(True , False)]
        fp = self.count_dict[(False, True)]
        fn = self.count_dict[(False, False)]
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp) if tn + fp > 0 else None
        f1_inv = (1 / precision + 1 / recall)
        f1     = 2 / f1_inv

        msg           = '\n'.join( [
        f"Accuracy    = {accuracy:.3f}"    ,
        f"Precision   = {precision:.3f}"   ,
        f"Recall      = {recall:.3f}"      ,
        f"Specificity = {specificity:.3f}" if specificity != None else f"Specificity = N/A",
        f"F1          = {f1:.3f}" ] )

        # place a text box in upper left in axes coords
        self.ax_l.text(0.0, 1.55, msg, transform=self.ax_l.transAxes, fontsize=18, verticalalignment='top', family = 'monospace')


    def show(self, filename = None): 
        self.fig, (self.ax_l, self.ax_r) = self.create_panels()

        self.plot_r(cat = (False, False), marker = "o", facecolor = 'none', edgecolor = '#E53935', label = 'False negative')
        self.plot_r(cat = (False, True ), marker = "o", facecolor = 'none', edgecolor = '#43A047', label = 'False positive')
        self.plot_l(cat = (True , False), marker = "o", facecolor = 'none', edgecolor = '#E53935', label = 'True negative')
        self.plot_l(cat = (True , True ), marker = "o", facecolor = 'none', edgecolor = '#43A047', label = 'True positive')
        self.report_metrics()
        self.plot_legend()

        plt.subplots_adjust(bottom = -0.1)

        if not isinstance(filename, str): 
            plt.show()
        else:
            # Set up drc...
            DRCPDF         = "pdfs"
            drc_cwd        = os.getcwd()
            prefixpath_pdf = os.path.join(drc_cwd, DRCPDF)
            if not os.path.exists(prefixpath_pdf): os.makedirs(prefixpath_pdf)

            # PDF doesn't need x index (hard to see anyway)...
            self.ax_l.set_xticks([])
            self.ax_r.set_xticks([])

            # Specify file...
            fl_pdf = f"{filename}.pdf"
            path_pdf = os.path.join(prefixpath_pdf, fl_pdf)

            # Export...
            ## plt.savefig(path_pdf, dpi = 100, bbox_inches='tight', pad_inches = 0)
            plt.savefig(path_pdf, dpi = 100)




def judger(dist, threshold):
    return True if dist < threshold else False




# File to analyze...
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

figsize = (16, 6)
disp_manager = VizRaw(res_dict = res_dict, figsize = figsize)
fl_pdf = f"alog.{id_log}"
disp_manager.show()
## disp_manager.show(filename = fl_pdf)
