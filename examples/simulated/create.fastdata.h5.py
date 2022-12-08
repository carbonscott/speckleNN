#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import h5py
import random

from deepprojection.plugins import PsanaImg
from deepprojection.utils   import split_dataset, set_seed

exp           = 'amo06516'
run           = '90'
mode          = 'idx'
detector_name = 'Camp.0:pnCCD.0'

psana_img = PsanaImg( exp           = exp,
                      run           = run,
                      mode          = mode,
                      detector_name = detector_name, )

seed = 0
random.seed(seed)

pdb = '1RWT'
pdb_similar = '6Q5U'
drc = f'skopi/pnccd.structure_similar.{pdb_similar}'

fl_h5_list = [ (f'{pdb}.1_hit.h5', 1, 180),
               (f'{pdb}.2_hit.h5', 2, 60),
               (f'{pdb}.3_hit.h5', 2, 60),
               (f'{pdb}.4_hit.h5', 2, 60), ]

hit_global_list = []
for fl_h5, label, num_hit in fl_h5_list:
    path_h5 = os.path.join(drc, fl_h5)
    with h5py.File(path_h5, 'r') as fh:
        hit_list = random.sample(range(len(fh.get('photons'))), num_hit)
        hit_global_list.extend([ (psana_img.get(0, fh.get('photons')[hit_idx]), label, (path_h5, str(hit_idx), str(label))) for hit_idx in hit_list])


import pickle
path_pickle = f'fastdata.h5/{pdb}.pnccd.pickle'
with open(path_pickle, 'wb') as handle:
    pickle.dump(hit_global_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
