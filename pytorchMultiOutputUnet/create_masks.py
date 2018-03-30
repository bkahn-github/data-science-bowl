import os
import glob
from tqdm import tqdm
from PIL import Image
from imageio import imwrite
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

def create_masks(root_folder, stage_number, stage_section, output_folder, subset=False):
    stage_folder = os.path.join(root_folder, 'stage' + stage_number + '_' + stage_section) 
    os.makedirs(stage_folder + '_masks', exist_ok=True)

    if subset:
        masks_folder = glob.glob(os.path.join(stage_folder, '*'))[:10]
    else:
        masks_folder = glob.glob(os.path.join(stage_folder, '*'))        

    for mask_folder in tqdm(masks_folder):
        mask_id = mask_folder.split('/')[-1]
        masks = []
        for mask in glob.glob(os.path.join(mask_folder, 'masks/*')):
            img = Image.open(mask)
            img = np.asarray(img)
            img = img / 255.0
            masks.append(img)
        total_mask = np.sum(masks, axis=0)

        mask_path = os.path.join(root_folder, 'stage' + stage_number +'_' + stage_section + '_masks', mask_id + '.png')        
        imwrite(mask_path, total_mask)

def create_contours(root_folder, stage_number, stage_section, output_folder, subset=False):
    stage_folder = os.path.join(root_folder, 'stage' + stage_number + '_' + stage_section) 
    os.makedirs(stage_folder + '_contours', exist_ok=True)

    if subset:
        masks_folder = glob.glob(os.path.join(stage_folder, '*'))[:10]
    else:
        masks_folder = glob.glob(os.path.join(stage_folder, '*'))        
    
    for mask_folder in tqdm(masks_folder):
        mask_id = mask_folder.split('/')[-1]
        masks = []
        for mask in glob.glob(os.path.join(mask_folder, 'masks/*')):
            img = Image.open(mask)
            img = np.asarray(img)
            img = img / 255.0

            img_contour = np.zeros_like(img).astype(np.uint8)
            _, contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img_contour, contours, -1, (255, 255, 255), 4)

            masks.append(img_contour)
        total_mask = np.where(np.sum(masks, axis=0) > 128., 255., 0.).astype(np.uint8)        

        contour_path = os.path.join(stage_folder + '_contours', mask_id + '.png')        
        imwrite(contour_path, total_mask)

def create_centers(root_folder, stage_number, stage_section, output_folder, subset=False):
    stage_folder = os.path.join(root_folder, 'stage' + stage_number + '_' + stage_section) 
    os.makedirs(stage_folder + '_centers', exist_ok=True)

    if subset:
        masks_folder = glob.glob(os.path.join(stage_folder, '*'))[:10]
    else:
        masks_folder = glob.glob(os.path.join(stage_folder, '*'))        
    
    for mask_folder in tqdm(masks_folder):
        mask_id = mask_folder.split('/')[-1]
        masks = []
        for mask in glob.glob(os.path.join(mask_folder, 'masks/*')):
            img = Image.open(mask)
            img = np.asarray(img)
            img = img / 255.0

            img_center = np.zeros_like(img).astype(np.uint8)
            x, y = ndimage.measurements.center_of_mass(img)
            cv2.Circle(img_center, (x, y), 4, (255, 255, 255), -1)

            masks.append(img_center)
        total_mask = np.where(np.sum(masks, axis=0) > 128., 255., 0.).astype(np.uint8)        

        center_path = os.path.join(stage_folder + '_centers', mask_id + '.png')        
        imwrite(center_path, total_mask)
