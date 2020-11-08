from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# resizing all images to 224 x 224
IMAGE_SIZE = [224, 224]

train_path = 'Datasets/Train'
valid_path = 'Datasets/Test'

#adding the preprocessing layer to the front of the VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

#don't train existing weights
for layer in vgg.layers:
    layer.trainable = False

#useful for getting number of classes
folders = glob('Datasets/Train/*')