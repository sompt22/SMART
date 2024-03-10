import os
import cv2

def mkdirs(d):
    os.makedirs(d, exist_ok=True)

# YOLO to MOTChallenge format conversion script
def yolo_to_mot(input, output):
    all_input_labels = f'{input}/labels_with_ids/train/'
    print(all_input_labels)
    all_input_images= f'{input}/images/train/'    
    print(all_input_images)

    seqs = os.listdir(all_input_labels)
    for seq in seqs:
        '''Inputs'''
        input_image_path = f'{all_input_images}/{seq}/img1/'
        input_label_path = f'{all_input_labels}/{seq}/img1/'
        '''Outputs'''
        output_image_path = f'{output}/train/{seq}/img1/'
        output_gt_path = f'{output}/train/{seq}/gt/'
        mkdirs(output_image_path)
        mkdirs(output_gt_path)      
        gt_txt = f'{output_gt_path}/gt.txt'

        labels = os.listdir(input_label_path)
        for label in labels:
            frame = int(label.split('.')[0])
            image = cv2.imread(f'{input_image_path}' + label.replace(".txt", ".jpg"))
            height, width, _ = image.shape
            cv2.imwrite(output_image_path + label.replace(".txt", ".jpg"), image)

            with open(f'{input_label_path}/{label}', 'r') as f:
                annotations = f.readlines()
            for annotation in annotations:
                class_id, track_id, xcenter, ycenter, bbox_width, bbox_height = annotation.split()

                bbox_abs_xcenter = float(xcenter)*width
                bbox_abs_ycenter = float(ycenter)*height
                bbox_abs_width = float(bbox_width)*width
                bbox_abs_height = float(bbox_height)*height
                              
                bbox_xmin = float(bbox_abs_xcenter) - float(bbox_abs_width) / 2    # int(float(xcenter) - float(bbox_width) / 2)
                bbox_ymin = float(bbox_abs_ycenter) - float(bbox_abs_height) / 2   #int(float(ycenter) - float(bbox_height) / 2)
                bbox_xmax = float(bbox_abs_xcenter) + float(bbox_abs_width) / 2    #int(float(xcenter) + float(bbox_width) / 2)
                bbox_ymax = float(bbox_abs_ycenter) + float(bbox_abs_height) / 2   #int(float(ycenter) + float(bbox_height) / 2)
                
                bbox_xmin = max(bbox_xmin, 0)
                bbox_ymin = max(bbox_ymin, 0)
                bbox_xmax = min(bbox_xmax, width - 1)
                bbox_ymax = min(bbox_ymax, height - 1)

                w = bbox_xmax - bbox_xmin
                h = bbox_ymax - bbox_ymin

                bbox_xmin = int(bbox_xmin) 
                bbox_ymin = int(bbox_ymin)
                w = int(w)
                h = int(h)

                with open(gt_txt, "a") as f:
                    label_str = '{},{},{},{},{},{},1,1,1.0\n'.format(frame, track_id, bbox_xmin, bbox_ymin, w, h)
                    f.write(label_str)
        print("{} finished!".format(seq))

# Usage example
input = '/home/fatih/phd/30042023/yolo/MOT17'
output = '/home/fatih/phd/30042023/yolo/yolo2mot_MOT17'

yolo_to_mot(input, output)
