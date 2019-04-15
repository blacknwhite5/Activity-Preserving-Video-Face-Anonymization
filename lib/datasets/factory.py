# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jason, Ross Girshick
# --------------------------------------------------------
"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.JHMDB import JHMDB
from datasets.DALY import DALY

import numpy as np

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

#  Dataset used in Privacy project
#  LEN_LIST = ['1', '5', '10']
#  MOD_LIST = ['RGB', 'FLOW']
MODI_LIST = ['raw', 'noise', 'blur-avg7x7', 'blur-down8x8', 'mask']

# JHMDB
for split in ['0', '1', '2']:
    for PHASE in ['train', 'test']:
        for modi in MODI_LIST:
            name = 'JHMDB_RGB_1_split_{}_{}_{}'.format(split, modi, PHASE)
            __sets[name] = (lambda image_set=name: JHMDB(image_set))

# DALY
for PHASE in ['train', 'test']:
    for modi in MODI_LIST:
        name = 'DALY_RGB_1_{}_{}'.format(modi, PHASE)
        __sets[name] = (lambda image_set=name: DALY(image_set))

# TODO: AVA

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()

def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
