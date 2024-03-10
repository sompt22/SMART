#import pafy
import cv2
import time
import os
import sys
from datetime import datetime
import shutil
import numpy as np

root_path = '/home/fatih/phd/mot_dataset/DIVOTrack/DIVO/images/test/Circle_View1'
outdir = '/home/fatih/Videos/test/'
os.mkdir(outdir)
annoTxt = open(f'{root_path}/gt/gt.txt', 'r')

lines = annoTxt.readlines()
lines = sorted(lines, key=lambda x:int(x.split(',')[0])) # sort by track_id
totalLine = len(lines)
colors = [tuple(map(int, color)) for color in np.random.randint(120, 250, (1000, 3))]

ims = os.listdir(f'{root_path}/img1')
ims.sort()
total_frames = len(ims)
bbox = 0
for i in range(1, total_frames+1):
    frame = cv2.imread(f'{root_path}/img1/{ims[i-1]}')    
    while  int(lines[bbox].split(',')[0]) == i: 
        id = int(lines[bbox].split(',')[1])
        x1 = float(lines[bbox].split(',')[2])
        y1 = float(lines[bbox].split(',')[3])
        x2 = float(lines[bbox].split(',')[4])
        y2 = float(lines[bbox].split(',')[5])
        if int(x1) < 0:
            x1 = '0'
        if int(x1) < 0:
            x1 = '0'

        x2 += x1
        y2 += y1
        color = colors[id]
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, str(id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        bbox += 1
        if bbox == totalLine:
            break
    cv2.imwrite(outdir+'img{}.jpg'.format(i), frame)

