import sys
import os
import keras
import time
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau  # Reference: https://keras.io/callbacks/
import h5py

batch_size = 32  # batch size
epoch_count = 400  # Number of epochs to train
image_shape = (32, 32)

# Load and augment training data
tr_datagen = image.ImageDataGenerator(
    featurewise_center=True,  # Boolean. Set input mean to 0 over the dataset, feature-wise
    samplewise_center=False,  # Boolean. Set each sample mean to 0
    featurewise_std_normalization=True,  # Boolean. Divide inputs by std of the dataset, feature-wise
    samplewise_std_normalization=False,  # Boolean. Divide each input by its std
    zca_whitening=False,  # Boolean. Apply ZCA whitening
    rotation_range=15,  # Int. Degree range for random rotations
    width_shift_range=0.12,  # Float. Range for random horizontal shifts
    height_shift_range=0.12,  # Float. Range for random vertical shifts
    shear_range=0.2,  # Float. Shear Intensity
    zoom_range=0.2,  # Float. Range for random zoom
    fill_mode='nearest',  # Points outside the boundaries of the input are filled according to the default nearest state
    horizontal_flip=True,  # Boolean. Randomly flip inputs horizontally
    vertical_flip=False)  # Boolean. Randomly flip inputs vertically

tr_generator = tr_datagen.flow_from_directory(
    './Heiro_train/',  # this is where the training data is
    target_size=image_shape,  # all images should be resized to 32x32
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical')

# validation data generation and preprocessing
val_datagen = image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rescale=None,
                                       rotation_range=15,
                                       width_shift_range=0.12,
                                       height_shift_range=0.12,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       fill_mode='nearest',
                                       horizontal_flip=True)
# this is a similar generator, for validation data
val_generator = val_datagen.flow_from_directory(
    './Heiro_val/',
    target_size=image_shape,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale')

# Load architecture
f = open('augmented_model_architecture.json', 'r')
model = model_from_json(f.read())
f.close()

# Load weights
model = load_model('theweights_extra.hdf5')
model.summary()

# Stochastic Gradient Descent optimizer.
sgd = SGD(lr=0.00001, momentum=0.7, decay=0.0001, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# reading current time to differentiate files in logs and submissions.
current = time.strftime("%c")

checkpointer = ModelCheckpoint(filepath="theweights_extra_part2.hdf5", verbose=1, save_best_only=True, monitor='val_loss')
tb = TensorBoard(log_dir='./logsKanji/' + current, histogram_freq=0, write_graph=True, write_images=False)

# Using reduce Learning rate on plateau to try to prevent saturation and overfitting by reducing the learning rate.
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=1e-6)

# Train model
model.fit_generator(
    tr_generator,
    samples_per_epoch=1240,  # amount of data we want to train on
    nb_epoch=epoch_count,
    validation_data=val_generator,
    nb_val_samples=200,  # amount of data we want to validate on
    callbacks=[tb, checkpointer])



