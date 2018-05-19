import torch

class Config():

    if torch.cuda.is_available():
        ROOT_FOLDER = '/content/.kaggle/competitions/data-science-bowl-2018/'
    else:
        ROOT_FOLDER = '/home/bilal/.kaggle/competitions/data-science-bowl-2018/'

    STAGE='1'

    IMGS_FOLDER = 'stage1_train'
    TARGETS_FOLDER = 'stage1_masks'

    SUBSET = False

    SHUFFLE = True
    BATCH_SIZE = 4
    NUM_WORKERS = 3

    KFOLDS = 6
    PATIENCE = 0
    EPOCHS = 10
    LR = 0.0001
    WEIGHTS = ''

    AUGMENT = True
    CLIP_LIMIT = 2
    GRID_SIZE = (8, 8)
    INVERT = 127
    RANDOM_CROP = 256
    FLIP_LR = 0.5
    FLIP_UD = 0.5
    ROTATE = 25

config = Config()