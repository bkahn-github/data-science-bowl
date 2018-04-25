import torch

class Config():

    if torch.cuda.is_available():
        ROOT_FOLDER = '/content/.kaggle/competitions/data-science-bowl-2018/'
    else:
        ROOT_FOLDER = '/home/bilal/.kaggle/competitions/data-science-bowl-2018/'

    STAGE='1'

    IMGS_FOLDER = 'stage1_train'

    MASKS_OUTPUT_FOLDER = 'stage1_masks'
    CONTOURS_OUTPUT_FOLDER = 'stage1_contours'
    CENTERS_OUTPUT_FOLDER = 'stage1_centers'

    SUBSET = False

    SHUFFLE = True
    BATCH_SIZE = 4
    NUM_WORKERS = 3

    SPLITS = 6
    PATIENCE = 0
    EPOCHS = 10
    WEIGHTS = ''

config = Config()