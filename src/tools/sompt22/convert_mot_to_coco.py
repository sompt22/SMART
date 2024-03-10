import os
import numpy as np
import json
import cv2

# Use the same script for MOT16
# DATA_PATH = '../../data/mot16/'
DATA_PATH = '/media/hdd4tb/faircentermot_dataset/SOMPT22/images/'
OUT_PATH = DATA_PATH + 'annotations/'
SPLITS = ['train','val','test']

if __name__ == '__main__':
  for split in SPLITS:
    data_path = DATA_PATH + f'{split}'
    out_path = OUT_PATH + '{}.json'.format(split)
    out = {'images': [], 'annotations': [], 
           'categories': [{'id': 1, 'name': 'pedestrain'}],
           'videos': [] , 'total_id': 0}
    seqs = os.listdir(data_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    tot_id = 0
    for seq in sorted(seqs):
      seq_path = '{}/{}/'.format(data_path, seq)
      print(seq_path)
      video_cnt += 1
      out['videos'].append({
        'id': video_cnt,
        'file_name': seq})        
      img_path = seq_path + 'img1/'
      print(img_path)
      ann_path = seq_path + 'gt/gt.txt'
      images = os.listdir(img_path)
      images.sort()
      #print(images)
      num_images = len([image for image in images if 'jpg' in image])
      img= cv2.imread(DATA_PATH+'/' + split +'/{}/img1/{}'.format(seq,images[0]))
      height, width, c = img.shape       
      for i in range(num_images):
        image_info = {'file_name': '{}/img1/{}'.format(seq, images[i]),
                      'width': height,
                      'height': width,
                      'id': image_cnt + i + 1,
                      'frame_id': i + 1,
                      'prev_image_id': image_cnt + i if i > 0 else -1,
                      'next_image_id': \
                        image_cnt + i + 2 if i < num_images - 1 else -1,
                      'video_id': video_cnt}
        out['images'].append(image_info)
      print('{}: {} images'.format(seq, num_images))
      if split != 'test':
        det_path = seq_path + 'det/det.txt'
        anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
        max_id = int(anns[:,1].max())
        print(f'seq: {seq}, max id: {int(anns[:,1].max())}')
        print(' {} ann images'.format(int(anns[:, 0].max())))
        for i in range(anns.shape[0]):
          frame_id = int(anns[i][0])
          track_id = int(anns[i][1])
          cat_id = int(anns[i][7])
          ann_cnt += 1
          category_id = 1
          ann = {'id': ann_cnt,
                'category_id': category_id,
                'image_id': image_cnt + frame_id,
                'track_id': track_id + tot_id,
                'bbox': anns[i][2:6].tolist(),
                'area': float(anns[i][4])*float(anns[i][5]),
                'iscrowd': 0,
                'conf': float(anns[i][6])}
          out['annotations'].append(ann)
        tot_id += max_id
        print(f'total id: {tot_id}')
      image_cnt += num_images
    print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
    out['total_id'] = tot_id
    json.dump(out, open(out_path, 'w'))
        
        

