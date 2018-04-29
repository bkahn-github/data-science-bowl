import os
import glob
from tqdm import tqdm
from PIL import Image
from imageio import imwrite
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

def get_contours(img):
    img_contour = np.zeros_like(img).astype(np.uint8)
    _, contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_contour, contours, -1, (255, 255, 255), 1)

    return img_contour

def create_masks(root_folder, stage_number, stage_section, output_folder, mode, subset=False):
    stage_folder = os.path.join(root_folder, 'stage' + stage_number + '_' + stage_section) 
    os.makedirs(stage_folder + '_' + mode, exist_ok=True)

    if subset:
        masks_folder = glob.glob(os.path.join(stage_folder, '*'))[:20]
    else:
        masks_folder = glob.glob(os.path.join(stage_folder, '*'))        
    
    for mask_folder in tqdm(masks_folder):
        mask_id = mask_folder.split('/')[-1]
        masks = []
        for mask in glob.glob(os.path.join(mask_folder, 'masks/*')):
            img = Image.open(mask)
            img = np.asarray(img)
            img = img / 255.0

            if mode == 'contours':
                img = get_contours(img)
            elif mode == 'masks':
                img = img

            masks.append(img)
        
        if mode == 'masks':
            total_mask = np.sum(masks, axis=0) * -1
        else:
            total_mask = np.where(np.sum(masks, axis=0) > 128., 255., 0.).astype(np.uint8) * -1 

        center_path = os.path.join(stage_folder + '_' + mode, mask_id + '.png')        
        imwrite(center_path, total_mask)
