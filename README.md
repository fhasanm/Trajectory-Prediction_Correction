# Trajectory Prediction Correction

 
## Datasets

Three datasets that were used can be downloaded from [Brain4cars](https://github.com/asheshjain399/ICCV2015_Brain4Cars), [CULane](https://xingangpan.github.io/projects/CULane.html) and [NGSIM](https://datahub.transportation.gov/stories/s/Next-Generation-Simulation-NGSIM-Open-Data/i5zb-xe34/) For Brain4Cars, videos are extracted into images with the fps=25 under each directory. The file name format is e.g. "image-0001.png". You can use the script `extract_frames.py` in `datasets/annotation` to extract images: Copy this file to the directory of "face_camera", and then run this script. For NGSIM, the raw data needs to be preprocessed from [preprocess_data.m](https://github.com/nachiket92/conv-social-pooling/blob/master/preprocess_data.m) into .m files. For CULane, download the dataset and extract them to `$CULANEROOT`. Create link to `data` directory.

```Shell
cd $CLRNET_ROOT
mkdir -p data
ln -s $CULANEROOT data/CULane
```

For CULane, you should have a structure like this:
```
$CULANEROOT/driver_xx_xxframe    # data folders x6
$CULANEROOT/laneseg_label_w16    # lane segmentation labels
$CULANEROOT/list                 # data lists
```



## Algorithms


### Lane Estimation

For training, run
```Shell
python main.py [configs/path_to_your_config] --gpus [gpu_num]
```
For example, run
```Shell
python main.py configs/clrnet/clr_resnet18_culane.py --gpus 0
```
For testing, run
```Shell
python main.py [configs/path_to_your_config] --[test|validate] --load_from [path_to_your_model] --gpus [gpu_num]
```
For example, run
```Shell
python main.py configs/clrnet/clr_dla34_culane.py --validate --load_from culane_dla34.pth --gpus 0
```

Currently, this code can output the visualization result when testing, just add `--view`.
We will get the visualization result in `work_dirs/xxx/xxx/visualization`.



### Intention Recognition

The dataset using 5-fold cross validaton. Run script ``n_fold_Brain4cars.py`` in directory ``datasets/annotation`` to split.

You can use the five ``.csv`` files in ``datasets/annotation`` and skip this step.

The main.py can be run on 3D-ResNet 50 to train the model. You can also continue training from the `.pth` in the directory. Some notes are:

1. ``root_path``: path to this project.

2. ``annotation_path`` : path to annotation directory in this project.

3. ``video_path``: path to image frames of driver videos.

4. ``pretrain_path``: path to the pretrained 3D ResNet 50 model.

5. ``n_fold``: is the number of the fold. Here, n_fold is from 0 to 4.

6. ``sample_duration``: the length of input vidoes. Here, 16 frames.

7. ``end_second``: the second before the maneuver. Here, end_second is from 1 to 5.

E.g. end_second = 3, frames which are 3 seconds (including the third second) before maneuver are given as input.

More details about other args, please refer to the ``opt.py``.



### Trajectory Prediction
There are two directories, one for a convolution-based network (Traj_Pred_conv), another for a transformer based network (Traj_Pred_transf). The training weights are saved in both files. The deployment code is also included. For further training, run main.py with training mode on. Validation scripts are also included. The conda environment needed to run the files is included in the xml. 

