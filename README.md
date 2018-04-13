# data-science-bowl

## Structure

### Google Drive

#### Config 1

Standard mask rcnn forked from  https://github.com/killthekitten/kaggle-ds-bowl-2018-baseline.
Trained from 12 to 136 epochs, cores from 0.25 to 0.36

#### Config 2
Same standard mask rcnn from Config 1 but with a modified configuration file from https://www.kaggle.com/c/data-science-bowl-2018/discussion/49805. Trained at default lr, 1e-3, 1e-4, and initialized on imagenet and coco. Scores from 0.29 to 0.0.412

#### Config 3
Same standard mask rcnn from Configs 1 and 2 but with a modified configuration file from a pytorch mask rcnn repository. Trained at default lr, 1e-4, and with a resnet101. Scores up to 0.425, the current best

#### Config 4
Modified mask rcnn forked from https://github.com/neptune-ml/open-solution-data-science-bowl-2018/tree/mask_rcnn_notebook. Trained at default lr, with imgaug, and image normalization. Scores up to 0.425, the current best. From 0.32 to 0.425, also the current best.

#### Config 5
A fork of the official matterport implementation from https://github.com/matterport/Mask_RCNN. Trained from 20 to 60 epochs. Results around 0.4-0.42

## Installation

### Install packages
```pip install -r data-science-bowl/requirements.txt``` or ```pip3 install -r data-science-bowl/requirements.txt```

### Create a .kaggle directory
```!mkdir ~/.kaggle```

### Create a `kaggle.json` file
```python
f = open(".kaggle/kaggle.json", "w")
f.write('{"username":"bkkaggle","key":[KEY]}')
f.close()
```

### Download the data 
`!kaggle competitions download -c data-science-bowl-2018` or `kaggle competitions download -c data-science-bowl-2018`

### unzip the data
```
!unzip ~/.kaggle/competitions/data-science-bowl-2018/stage1_train.zip -d ~/.kaggle/competitions/data-science-bowl-2018/train
!unzip ~/.kaggle/competitions/data-science-bowl-2018/stage1_test.zip -d ~/.kaggle/competitions/data-science-bowl-2018/test
!unzip ~/.kaggle/competitions/data-science-bowl-2018/stage1_train_labels.csv.zip -d ~/.kaggle/competitions/data-science-bowl-2018/labels
```

Or 
```
unzip ~/.kaggle/competitions/data-science-bowl-2018/stage1_train.zip -d ~/.kaggle/competitions/data-science-bowl-2018/train
unzip ~/.kaggle/competitions/data-science-bowl-2018/stage1_test.zip -d ~/.kaggle/competitions/data-science-bowl-2018/test
unzip ~/.kaggle/competitions/data-science-bowl-2018/stage1_train_labels.csv.zip -d ~/.kaggle/competitions/data-science-bowl-2018/labels
```

### Change working directory (For jupyter notebooks)

```python 
import os
os.chdir('data-science-bowl/')
```
