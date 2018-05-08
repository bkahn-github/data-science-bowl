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
    img = skimage.morphology.binary_dilation(img, selem=np.ones((5,5))).astype(np.uint8)
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
        masks_with_edges = np.zeros(size)

        for mask in glob.glob(os.path.join(mask_folder, 'masks/*')):
            img = Image.open(mask)
            img = np.asarray(img)
            img = img / 255.0

            img_with_edges = get_edges(img)
            
            masks = np.add(masks, img)
            masks_with_edges = np.add(masks_with_edges, img_with_edges)
        
        target = np.zeros((size[0], size[1], 3))
        
        target[:,:,0] = masks == 1
        target[:,:,1] = masks_with_edges == 2
        target[:,:,2] = masks == 0
        
        target *= 255
        target = target.astype(np.uint8)

        output_path = os.path.join(stage_folder + '_' + mode, mask_id + '.png')        
        imwrite(output_path, target)
