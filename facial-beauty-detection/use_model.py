from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image

import numpy as np
import os
import pandas as pd
import PIL

# Define paths
local_path = os.path.abspath('/home/rexcolt/GitHub/machine-learning-projects/facial-beauty-detection')
if not os.path.isdir(local_path):
    local_path = None

dataset_path = os.path.relpath('datasets/SCUT-FBP5500_v2')
if local_path:
    dataset_path = os.path.join(local_path, dataset_path)

csv_file_path = os.path.join(dataset_path, 'train_test_files', 'All_labels.txt')
df = pd.read_csv(csv_file_path, header=None, names=['filename', 'rating'], sep=' ')
images_path = os.path.join(dataset_path, 'Images')

models_path = os.path.relpath('models')
if local_path:
    models_path = os.path.join(local_path, models_path)

# Load trained model
model = load_model(os.path.join(models_path, 'beauty_model_untuned.h5'))

def predict_from_img_path(img_filename='nicolas_cage.jpg'):
    img_size = 224
    img = image.load_img(os.path.join(local_path, 'downloaded_images', img_filename), target_size=(img_size, img_size))

    # Preprocess image
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]
    print(f'\n{img_filename} - Predicted beauty: ', prediction)

predict_from_img_path()