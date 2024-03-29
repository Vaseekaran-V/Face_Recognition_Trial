from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model, load_model
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

#layers that are going to be trained
x = Flatten()(vgg.output)
#x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation = 'softmax')(x)

#creating a model
model = Model(inputs = vgg.input, outputs = prediction)

#model summary
model.summary()

#choosing cost and optimization
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'Datasets/Train',
    target_size=(224,224),
    batch_size=32,
    class_mode = 'categorical'
)

test_set = test_datagen.flow_from_directory(
    'Datasets/Test',
    target_size=(224,224),
    batch_size=32,
    class_mode = 'categorical'
)

#fitting the model
r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

#displaying the loss and accuracies
plt.plot(r.history['loss'], label = 'train loss')
plt.plot(r.history['val_loss'], label = 'val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_Loss')

plt.plot(r.history['acc'], label = 'train accuracy')
plt.plot(r.history['val_acc'], label = 'val accuracy')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

#saving the model