import os
import numpy as np
import json
import cv2

# Use the same script for MOT16
DATA_PATH = '/home/fatih/phd/SMART/data/mot17/images/'
OUT_PATH = DATA_PATH + 'annotations/'
SPLITS = ['train']

# Flag to split the sequences and annotations
SPLIT_SEQUENCE_IN_HALF = True  # Set to False if you don't want to split the sequences

if __name__ == '__main__':
    for split in SPLITS:
        data_path = DATA_PATH + f'{split}'
        out_train_path = OUT_PATH + 'train_half.json'
        out_val_path = OUT_PATH + 'val_half.json'

        out_train = {'images': [], 'annotations': [], 
                     'categories': [{'id': 1, 'name': 'pedestrian'}],
                     'videos': [], 'total_id': 0}
                     
        out_val = {'images': [], 'annotations': [], 
                   'categories': [{'id': 1, 'name': 'pedestrian'}],
                   'videos': [], 'total_id': 0}

        seqs = os.listdir(data_path)
        image_cnt_train = 0
        image_cnt_val = 0
        ann_cnt_train = 0
        ann_cnt_val = 0
        video_cnt = 0
        tot_id_train = 0
        tot_id_val = 0
        
        for seq in sorted(seqs):
            if "FRCNN" not in seq:
              continue
            seq_path = '{}/{}/'.format(data_path, seq)
            print(f"Processing sequence: {seq_path}")
            video_cnt += 1
            
            # Append video information for both training and validation
            out_train['videos'].append({'id': video_cnt, 'file_name': seq})
            out_val['videos'].append({'id': video_cnt, 'file_name': seq})
            
            img_path = seq_path + 'img1/'
            ann_path = seq_path + 'gt/gt.txt'
            images = os.listdir(img_path)
            images.sort()
            num_images = len([image for image in images if 'jpg' in image])
            
            half_num_images = num_images // 2
            img = cv2.imread(DATA_PATH + '/' + split + '/{}/img1/{}'.format(seq, images[0]))
            height, width, c = img.shape
            
            # Paths for split annotations
            half_train_path = seq_path + 'gt/half_train.txt'
            half_val_path = seq_path + 'gt/half_val.txt'
            
            # Open the new files for half_train and half_val
            half_train_file = open(half_train_path, 'w')
            half_val_file = open(half_val_path, 'w')

            # Process first half for training
            for i in range(half_num_images):
                image_info_train = {'file_name': '{}/img1/{}'.format(seq, images[i]),
                                    'width': width,
                                    'height': height,
                                    'id': image_cnt_train + i + 1,
                                    'frame_id': i + 1,
                                    'prev_image_id': image_cnt_train + i if i > 0 else -1,
                                    'next_image_id': image_cnt_train + i + 2 if i < half_num_images - 1 else -1,
                                    'video_id': video_cnt}
                out_train['images'].append(image_info_train)

            # Process second half for validation
            for i in range(half_num_images, num_images):
                image_info_val = {'file_name': '{}/img1/{}'.format(seq, images[i]),
                                  'width': width,
                                  'height': height,
                                  'id': image_cnt_val + i + 1 - half_num_images,
                                  'frame_id': i + 1,
                                  'prev_image_id': image_cnt_val + i - half_num_images if i > half_num_images else -1,
                                  'next_image_id': image_cnt_val + i + 2 - half_num_images if i < num_images - 1 else -1,
                                  'video_id': video_cnt}
                out_val['images'].append(image_info_val)
                
            print('{}: {} images ({} for train, {} for val)'.format(seq, num_images, half_num_images, num_images - half_num_images))
            
            if split != 'test':
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                max_id = int(anns[:, 1].max())
                
                # Process annotations for training (first half of the images)
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    track_id = int(anns[i][1])
                    if frame_id <= half_num_images:
                        ann_cnt_train += 1
                        ann_train = {'id': ann_cnt_train,
                                     'category_id': 1,
                                     'image_id': image_cnt_train + frame_id,
                                     'track_id': track_id + tot_id_train,
                                     'bbox': anns[i][2:6].tolist(),
                                     'area': float(anns[i][4]) * float(anns[i][5]),
                                     'iscrowd': 0,
                                     'conf': float(anns[i][6])}
                        out_train['annotations'].append(ann_train)
                        
                        # Write to half_train.txt
                        half_train_file.write(','.join([str(int(x)) if x.is_integer() else str(x) for x in anns[i]]) + '\n')
                        
                tot_id_train += max_id
                
                # Process annotations for validation (second half of the images)
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    track_id = int(anns[i][1])
                    if frame_id > half_num_images:
                        ann_cnt_val += 1
                        ann_val = {'id': ann_cnt_val,
                                   'category_id': 1,
                                   'image_id': image_cnt_val + frame_id - half_num_images,
                                   'track_id': track_id + tot_id_val,
                                   'bbox': anns[i][2:6].tolist(),
                                   'area': float(anns[i][4]) * float(anns[i][5]),
                                   'iscrowd': 0,
                                   'conf': float(anns[i][6])}
                        out_val['annotations'].append(ann_val)
                        
                        # Write to half_val.txt
                        half_val_file.write(','.join([str(int(x)) if x.is_integer() else str(x) for x in anns[i]]) + '\n')
                        
                tot_id_val += max_id

            # Increment image counters
            image_cnt_train += half_num_images
            image_cnt_val += num_images - half_num_images

            # Close the files after writing
            half_train_file.close()
            half_val_file.close()

        print('Saving train_half.json and val_half.json...')
        json.dump(out_train, open(out_train_path, 'w'))
        json.dump(out_val, open(out_val_path, 'w'))
