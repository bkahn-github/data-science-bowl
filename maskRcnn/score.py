import model as modellib
import cv2
import os
import numpy as np
from tqdm import tqdm
from skimage.morphology import label
from skimage.transform import resize
import imageio

from inference_config import inference_config
from metrics import iou
import functions as f
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = os.path.join(ROOT_DIR, "model.h5")
# model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

train_path = './stage1_train/'
test_path = './stage1_test/'

train_ids = next(os.walk(train_path))[1][:10]
test_ids = next(os.walk(test_path))[1][:10]

train_masks = []
for image_id in tqdm(train_ids):  
  masks = np.zeros((256, 256, 1), dtype=np.bool)
  mask_files = next(os.walk(train_path + image_id + '/masks/'))[2]
  
  for mask in mask_files:
    mask = cv2.imread(train_path + '/' + image_id + '/masks/' + mask)
    if len(mask.shape) > 2:
      mask = mask[:,:,0]
    mask = (resize(mask, (256, 256), mode='constant', preserve_range=True)).reshape(256, 256, 1)
    masks = np.maximum(masks, mask)  

  train_masks.append(masks)

train_preds = []
for image_id in tqdm(train_ids):
  image_path = os.path.join('stage1_train', image_id, 'images', image_id + '.png')

  original_image = cv2.imread(image_path)
  results = model.detect([original_image], verbose=0)
  r = results[0]

  masks = r['masks']
  train_preds.append(masks)
                
predicted_masks = []
for i, train_pred in enumerate(train_preds):
  img = np.zeros((256, 256, 1), dtype=np.bool)

  for j in range(train_preds[i].shape[-1]):
    mask = train_preds[i][:,:,j]
    mask = np.expand_dims(resize(mask, (256, 256), mode='constant', preserve_range=True), axis=-1)

    img = np.maximum(img, mask)    

  predicted_masks.append(img)
                         
preds = np.zeros((len(predicted_masks), 256, 256, 1))
masks = np.zeros((len(train_masks), 256, 256, 1))

for i, pred in enumerate(predicted_masks):
  preds[i] = pred * 255.0/pred.max()
  
for i, mask in enumerate(train_masks):
  masks[i] = mask
                         
ious = []
for i in range(len(predicted_masks)):
  ious.append(iou(label(preds[i] > 0.5), label(masks[i] > 0.5))[0])
  
print(np.asarray(ious).mean())