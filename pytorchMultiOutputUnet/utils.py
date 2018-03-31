import os
import glob

from config import root_folder, stage, imgs_folder, masks_output_folder, contours_output_folder, centers_output_folder, subset

def get_ids():
    if subset:
        ids = glob.glob(os.path.join(root_folder, 'stage' + stage + '_train', '*'))[:10]
    else:
        ids = glob.glob(os.path.join(root_folder, 'stage' + stage + '_train', '*'))

    ids = [id.split('/')[-1] for id in ids]
    return ids

def get_path(id):
    img_path = os.path.join(root_folder, 'stage' + stage + '_train', id, 'images', id + '.png')

    mask_path = os.path.join(root_folder, 'stage' + stage + '_train_masks', id + '.png')
    contour_path = os.path.join(root_folder, 'stage' + stage + '_train_contours', id + '.png')
    center_path = os.path.join(root_folder, 'stage' + stage + '_train_centers', id + '.png')

    return img_path, mask_path, contour_path, center_path
