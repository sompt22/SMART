import cv2
import time
import os
import sys
from datetime import datetime
import shutil
import numpy as np

colors = [tuple(map(int, color)) for color in np.random.randint(120, 250, (1000, 3))]
root_path = '/home/fatih/phd/mot_dataset/DIVOTrack/DIVO'
img_path = root_path + '/images/test'
seqs = os.listdir(img_path)
seqs = sorted(seqs)
for seq in seqs:
    print(seq)
    root_img = img_path + '/' + seq + '/img1/'
    ims = os.listdir(root_img)
    ims = sorted(ims)
    size = (width, height) = (1920, 1080)
    out = cv2.VideoWriter(f'/home/fatih/Desktop/{seq}.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
    for im in ims:
        frm = int(im.split('.')[0].split('_')[2])
        im_path = root_img + im
        txt_path = im_path.replace('images', 'labels_with_ids').replace('jpg', 'txt')  
        img = cv2.imread(im_path)
        width, height = img.shape[1], img.shape[0]      
        with open(txt_path) as f:
            lines = f.readlines()
            for line in lines:
                pars = line.split(' ')
                classN = pars[0]
                track_id = pars[1]
                c1 = float(pars[2])*width
                c2 = float(pars[3])*height
                imwidth = float(pars[4])*width
                imheight = float(pars[5])*height
                x1 = int(c1 - imwidth/2)
                y1 = int(c2 - imheight/2)
                x2 = int(c1 + imwidth/2)
                y2 = int(c2 + imheight/2)
                color = colors[int(classN)]
                cv2.putText(img,track_id, (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 1, 1, 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)       
        out.write(img)
out.release()        
        #cv2.waitKey(500)
        # closing all open windows
        #cv2.destroyAllWindows()
   