from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import sys
import subprocess

from ..generic_dataset import GenericDataset

class CrowdHuman(GenericDataset):
  num_classes = 1
  num_joints = 17
  default_resolution = [544, 960]
  max_objs = 128
  class_name = ['person']
  cat_ids = {1: 1}

  @staticmethod
  def _resolve_img_dir(data_dir, split):
    candidates = [
      os.path.join(data_dir, '{}'.format(split)),
      os.path.join(data_dir, 'CrowdHuman_{}'.format(split), 'Images'),
      os.path.join(data_dir, 'CrowdHuman_{}'.format(split.capitalize()), 'Images')
    ]
    for candidate in candidates:
      if os.path.exists(candidate):
        return candidate
    return candidates[0]

  def __init__(self, opt, split):
    super(CrowdHuman, self).__init__()
    data_dir = os.path.join(opt.data_dir, 'crowdhuman')
    img_dir = self._resolve_img_dir(data_dir, split)
    ann_path = os.path.join(data_dir, 'annotations', 
      '{}.json').format(split)

    print('==> initializing CityPersons {} data.'.format(split))

    self.images = None
    # load image list and coco
    super(CrowdHuman, self).__init__(opt, split, ann_path, img_dir)

    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def _save_results(self, records, fpath):
    with open(fpath,'w') as fid:
      for record in records:
        line = json.dumps(record)+'\n'
        fid.write(line)
    return fpath

  def convert_eval_format(self, all_bboxes):
    detections = []
    person_id = 1
    for image_id in all_bboxes:
      if type(all_bboxes[image_id]) != type({}):
        # newest format
        dtboxes = []
        for j in range(len(all_bboxes[image_id])):
          item = all_bboxes[image_id][j]
          if item['class'] != person_id:
            continue
          bbox = item['bbox']
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          bbox_out  = list(map(self._to_float, bbox[0:4]))
          detection = {
              "tag": 1,
              "box": bbox_out,
              "score": float("{:.2f}".format(item['score']))
          }
          dtboxes.append(detection)
      img_info = self.coco.loadImgs(ids=[image_id])[0]
      file_name = img_info['file_name']
      detections.append({'ID': file_name[:-4], 'dtboxes': dtboxes})
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    self._save_results(self.convert_eval_format(results),
                       '{}/results_crowdhuman.odgt'.format(save_dir))
  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    ret = subprocess.run([
      sys.executable, 'tools/crowdhuman_eval/demo.py',
      '../data/crowdhuman/annotation_val.odgt',
      '{}/results_crowdhuman.odgt'.format(save_dir),
    ]).returncode
    if ret != 0:
      print('Crowdhuman evaluation not setup!')
