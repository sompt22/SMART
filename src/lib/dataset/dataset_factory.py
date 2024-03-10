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

dataset_factory = {
  'custom': CustomDataset,
  'coco': COCO,
  'coco_hp': COCOHP,
  'mot': MOT,
  'crowdhuman': CrowdHuman,
  'divo': DIVO,
  'mot20': MOT20,
  'sompt22': SOMPT22,
  'sompt22-train-sim': SOMPT22_TRAIN_SIM
}


def get_dataset(dataset):
  return dataset_factory[dataset]
