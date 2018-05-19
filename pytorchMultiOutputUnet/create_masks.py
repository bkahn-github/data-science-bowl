import os
import glob
import numpy as np
from tqdm import tqdm

import cv2
import skimage
import skimage.morphology
from imageio import imwrite

def get_edges(img):
    img = skimage.morphology.binary_dilation(img, selem=np.ones((5,5))).astype(np.uint8)
    return img
  
def get_sizes(mask_folder):
    mask = glob.glob(os.path.join(mask_folder, 'masks/*'))[0]
    mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

    return mask.shape

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
        masks_with_edges = np.zeros(size)

        for mask in glob.glob(os.path.join(mask_folder, 'masks/*')):
            img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
            img = img / 255.0

            img_with_edges = get_edges(img)
            
            masks = np.add(masks, img)
            masks_with_edges = np.add(masks_with_edges, img_with_edges)
        
        mask = np.zeros((size[0], size[1], 3))
        
        mask[:,:,0] = masks == 1
        mask[:,:,1] = masks_with_edges == 2
        mask[:,:,2] = masks == 0
        
        mask *= 255
        mask = mask.astype(np.uint8)

        output_path = os.path.join(stage_folder + '_' + mode, mask_id + '.png')        
        imwrite(output_path, mask)
