# Getting Started

This document provides tutorials to train and evaluate SMaRT. Before getting started, make sure you have finished [installation](INSTALL.md) and [dataset setup](DATA.md).

## Benchmark evaluation


### MOT20 Tracking

To test the tracking performance on MOT20 with our pretrained model, run

~~~
 python test.py tracking,embedding --exp_id mot20 --dataset mot20 --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model ../models/mot20.pth
~~~

### DIVOTrack Tracking

To test the tracking performance on DIVOTrack with our pretrained model, run

~~~
 python test.py tracking,embedding --exp_id divo --dataset divo --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model ../models/divo.pth
~~~

### SOMPT22 Tracking

To test the tracking performance on DIVOTrack with our pretrained model, run

~~~
 python test.py tracking,embedding --exp_id sompt22 --dataset sompt22 --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model ../models/sompt22.pth
~~~



## Training
We have packed all the training scripts in the [experiments](../experiments) folder. The number of GPUs for each experiment can be found in the scripts and the model zoo. If the training is terminated before finishing, you can use the same command with `--resume` to resume  raining. It will found the latest model with the same `exp_id`. Some experiments rely on pretraining on another model. In this case, download the pretrained model from our model zoo or train that model first.