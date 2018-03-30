import os
import glob
from tqdm import tqdm
from PIL import Image
from imageio import imwrite
import numpy as np
import matplotlib.pyplot as plt

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

# def create_contours(root_folder, stage_folder, output_folder):

# def create_centers(root_folder, stage_folder, output_folder):
