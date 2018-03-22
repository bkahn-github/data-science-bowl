import keras
import numpy as np

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.applications import ResNet50
model = ResNet50(weights='imagenet')

img = '/Users/bk/desktop/Machine-Learning/projects/[img name].jpg'

img = image.load_img(img, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])
