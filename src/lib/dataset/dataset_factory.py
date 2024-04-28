from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

from .datasets.coco import COCO
from .datasets.coco_hp import COCOHP
from .datasets.mot import MOT
from .datasets.crowdhuman import CrowdHuman
from .datasets.custom_dataset import CustomDataset
from .datasets.divo import DIVO
from .datasets.mot20 import MOT20
from .datasets.sompt22 import SOMPT22
from .datasets.sompt22_train_sim import SOMPT22_TRAIN_SIM
from .datasets.sompt22_train_sim_noocc import SOMPT22_TRAIN_SIM_NOOCC
from .datasets.sompt22_train_paper import SOMPT22_TRAIN_PAPER
from .datasets.sompt22_train_10secs_occlusion import SOMPT22_TRAIN_10SECS_OCCLUSION
from .datasets.sompt22_train_10secs_clear import SOMPT22_TRAIN_10SECS_CLEAR

dataset_factory = {
  'custom': CustomDataset,
  'coco': COCO,
  'coco_hp': COCOHP,
  'mot': MOT,
  'crowdhuman': CrowdHuman,
  'divo': DIVO,
  'mot20': MOT20,
  'sompt22': SOMPT22,
  'sompt22-train-sim': SOMPT22_TRAIN_SIM,
  'sompt22-train-sim-noocc': SOMPT22_TRAIN_SIM_NOOCC,
  'sompt22-train-paper': SOMPT22_TRAIN_PAPER,
  'sompt22-train-10secs-occlusion': SOMPT22_TRAIN_10SECS_OCCLUSION,
  'sompt22-train-10secs-clear': SOMPT22_TRAIN_10SECS_CLEAR
}


def get_dataset(dataset):
  return dataset_factory[dataset]
