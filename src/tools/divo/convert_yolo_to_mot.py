import os
import cv2

def mkdirs(d):
    os.makedirs(d, exist_ok=True)

# YOLO to MOTChallenge format conversion script
def yolo_to_mot(input, output, split):
    #all_input_labels = f'{input}/labels_with_ids/test/'
    #print(all_input_labels)
    all_input_images= f'{input}/images/{split}'    
    print(all_input_images)

    seqs = os.listdir(all_input_images)
    line_list =[]
    seqs.sort()
    for seq in seqs:
        '''Inputs'''
        input_image_path = f'{all_input_images}/{seq}/img1/'
        images = os.listdir(input_image_path)
        images.sort()
        numImgs = len(images)
        '''Outputs'''
        output_path = f'{output}/{split}/{seq}/'
        output_image_path = output_path + 'img1/'
        output_gt_path = output_path + 'gt'
        #mkdirs(output_image_path)
        mkdirs(output_gt_path)      
        gt_txt = f'{output_gt_path}/gt.txt'       
        line_list = []
        for image in images:
            frame = image.split('.')[0] 
            frame = int(frame.split('_')[2])
            if numImgs < frame:
                print("seq: ,frame: ", seq,frame)             
            if(frame % 10 == 1):
                print("frame: ", frame)
            img =  input_image_path + image
            label =  img.replace("images","labels_with_ids").replace(".jpg", ".txt")
            
            img1 = cv2.imread(img)
            height, width, _ = img1.shape
            #cv2.imwrite(output_image_path + image, img1)   
    
            with open(f'{label}', 'r') as f:
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
                
                label_str = '{},{},{},{},{},{},1,1,1.0\n'.format(frame, track_id, bbox_xmin, bbox_ymin, w, h)
                line_list.append(label_str) 
                
        line_list = sorted(line_list, key=lambda x:int(x.split(',')[1])) # sort by track_id        
        with open(gt_txt, "w") as fic:
            fic.writelines([line for line in line_list]) # write each line                    
        print("{} finished!".format(seq))                
        """   
        with open(gt_txt, "a") as f:
            label_str = '{},{},{},{},{},{},1,1,1.0\n'.format(frame, track_id, bbox_xmin, bbox_ymin, w, h)
            f.write(label_str)
        """
        
# Usage example
input_ = '/home/fatih/phd/mot_dataset/DIVOTrack/DIVO'
output = '/home/fatih/phd/mot_dataset/DIVOTrack/DIVO/test-anno'

yolo_to_mot(input_, output,"test")