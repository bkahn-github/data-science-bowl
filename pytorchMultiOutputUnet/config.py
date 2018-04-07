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

    SUBSET = True

    BATCH_SIZE = 4
    SHUFFLE = True
    NUM_WORKERS = 4

config = Config()