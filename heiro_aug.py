import keras
import numpy as np
import time

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau     # Reference: https://keras.io/callbacks/
from keras.applications import *
from keras.preprocessing import image
from keras.models import Model, Sequential , load_model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.constraints import maxnorm
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.optimizers import Adam, SGD

current = time.strftime("%c")

batch_size = 32

tr_datagen = image.ImageDataGenerator(                
    rotation_range = 5,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = False)

tr_generator = tr_datagen.flow_from_directory(
        './Heiro_train/',  # this is the target directory
        target_size=(28,28),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode = 'grayscale')

val_datagen = image.ImageDataGenerator(featurewise_center = False,featurewise_std_normalization = False,rescale = None)

# this is a similar generator, for validation data
val_generator = val_datagen.flow_from_directory(
        './Heiro_val/',
        target_size=(28,28),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode = 'grayscale')

if K.image_data_format() == 'channels_first':
    input_shape = (1, 28, 28)
else:
    input_shape = (28, 28, 1)
    
############## Creating Model ################

# initialize the model
##model = Sequential()
##
### first set of CONV => RELU => POOL
##model.add(Convolution2D(20, (5, 5), padding="same",
##        input_shape=(1, 32, 32)))
##model.add(Activation("relu"))
##model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
##
### second set of CONV => RELU => POOL
##model.add(Convolution2D(50, (5, 5), padding="same"))
##model.add(Activation("relu"))
##model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
##
### set of FC => RELU layers
##model.add(Flatten())
##model.add(Dense(500))
##model.add(Activation("relu"))
##
### softmax classifier
##model.add(Dense(10))
##model.add(Activation("softmax"))

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(40, activation='softmax'))

##model.load_weights("mnist.hdf5", by_name = True) 

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
# for layer in base_model.layers:
#      layer.trainable = False

sgd = SGD(lr=0.001, momentum=0.6, decay=0.000001, nesterov=True)

model.compile(loss = 'categorical_crossentropy', optimizer = sgd,  metrics = ['accuracy'])
model.summary()


# Training Model

checkpointer = ModelCheckpoint(filepath = "heiro_aug_scratch.hdf5", verbose = 1, save_best_only = True, monitor = 'val_loss')   
tb = TensorBoard(log_dir='./heiro_logs/'+ current, histogram_freq=0, write_graph=True, write_images=False)

#reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 5, min_lr = 1e-6)

history = model.fit_generator(
        tr_generator,
        steps_per_epoch=1200,
        epochs= 20,
        validation_data=val_generator,
        validation_steps=120,
        callbacks = [tb, checkpointer])




score = model.evaluate_generator(val_generator)
print('Test score:', score[0])
print('Test accuracy:', score[1])


