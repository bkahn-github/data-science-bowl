import torch

class Config():

    if torch.cuda.is_available():
        ROOT_FOLDER = '/content/.kaggle/competitions/data-science-bowl-2018/'
    else:
        ROOT_FOLDER = '/home/bilal/.kaggle/competitions/data-science-bowl-2018/'

    STAGE='1'

    IMGS_FOLDER = 'stage1_train'

    MASKS_OUTPUT_FOLDER = 'stage1_masks'
    EDGES_OUTPUT_FOLDER = 'stage1_edges'
    BACKGROUNDS_OUTPUT_FOLDER = 'stage1_backgrounds'

    SUBSET = False

    SHUFFLE = True
    BATCH_SIZE = 4
    NUM_WORKERS = 3

    KFOLDS = 6
    PATIENCE = 0
    EPOCHS = 10
    LR = 0.0001
    WEIGHTS = ''

config = Config()