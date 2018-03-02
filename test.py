


# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 08:10:00 2018

@author: GITESH
"""
###
from keras.datasets import mnist
from keras.models import Sequential
from sklearn.utils import shuffle
import numpy as np
from keras.optimizers import *
from keras.preprocessing.image import *
from keras import layers
from keras.layers import Input, Add, Dense,Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.applications import *
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.initializers import glorot_uniform
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.backend as K
K.set_image_data_format('channels_last')
from skimage.transform import rotate
from sklearn.feature_extraction import image
from skimage import io,color
from skimage.transform import resize
from numpy import array
import json
#from data_generator import read_benign,read_insitu,read_invasive,read_normal,train_generator, validation_generator,test_generator
model=vgg19.VGG19(include_top=False,input_shape=(224,224,3),weights='imagenet')
X=Flatten()(model.layers[-1].output)
X = Dense(4096, activation='relu', name='fc1')(X)
X = Dense(4096, activation='relu', name='fc2')(X)
X = Dense(4, activation='softmax', name='predictions')(X)
model = Model(inputs=model.input, outputs=X)
model = Model(inputs=model.input, outputs=X)
model.summary()
train_datagen = ImageDataGenerator( rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'Train',
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(
        'Validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
test_generator=test_datagen.flow_from_directory(
        'Test',
        target_size=(224,224), 
        batch_size=32,
        class_mode='categorical')
print(train_generator.class_indices)
print(test_generator.class_indices)
print(validation_generator.class_indices)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
preds = model.evaluate_generator(train_generator, steps=3)
print(preds)
print ("Loss = " + str(preds[0]))
print ("Train Accuracy = " + str(preds[1]))
preds = model.evaluate_generator(validation_generator, steps=3)
print(preds)
print ("Loss = " + str(preds[0]))
print ("Validation Accuracy = " + str(preds[1]))
preds = model.evaluate_generator(test_generator, steps=3)
print(preds)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
'''

batch_input_shape= Input(shape=(224,224,3),name = 'image_input')
X=model(batch_input_shape)

#Use the generated model 
print(X.shape)
#Add the fully-connected layers 
X = Flatten(name='flatten')(X)
X = Dense(4096, activation='relu', name='fc1')(X)
X = Dense(4096, activation='relu', name='fc2')(X)
X = Dense(4, activation='softmax', name='predictions')(X)
model = Model(inputs=batch_input_shape, outputs=X)

model.summary()

'''
'''
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()          
###
'''
print("DONE")