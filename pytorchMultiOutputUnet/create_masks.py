import os
import glob
from tqdm import tqdm
from PIL import Image
from imageio import imwrite
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import skimage.morphology

def get_edges(img):
    img = skimage.morphology.binary_dilation(img).astype(np.uint8)
    return img
  
def get_sizes(mask_folder):
    mask = glob.glob(os.path.join(mask_folder, 'masks/*'))[0]
    img = Image.open(mask)
    img = np.asarray(img)

    return img.shape

def create_masks(root_folder, stage_number, stage_section, output_folder, mode, subset=False):
    stage_folder = os.path.join(root_folder, 'stage' + stage_number + '_' + stage_section) 
    os.makedirs(stage_folder + '_' + mode, exist_ok=True)

    if subset:
        masks_folder = glob.glob(os.path.join(stage_folder, '*'))[:20]
    else:
        masks_folder = glob.glob(os.path.join(stage_folder, '*'))        
    
    for mask_folder in tqdm(masks_folder):
        mask_id = mask_folder.split('/')[-1]

        size = get_sizes(mask_folder)
        masks = np.zeros(size)

        for mask in glob.glob(os.path.join(mask_folder, 'masks/*')):
            img = Image.open(mask)
            img = np.asarray(img)
            img = img / 255.0

            img = get_edges(img)
            masks = np.add(masks, img)
        
        if mode == 'masks':
            masks[masks != 1] = 0
        elif mode == 'edges':
            masks[masks == 1] = 0
            masks[masks > 1] = 1
        elif mode == 'backgrounds':
            masks[masks != 1] = 0
            masks = (masks == 0).astype(np.uint8)

        output_path = os.path.join(stage_folder + '_' + mode, mask_id + '.png')        
        imwrite(output_path, masks)
