import keras
import numpy as np
import time
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau     # Reference: https://keras.io/callbacks/
from keras.applications import *
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.constraints import maxnorm
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.optimizers import Adam, SGD

batch_size = 20  # batch size
epoch_count = 400  # Number of epochs to train
image_shape = (32, 32)

# Load and augment training data
tr_datagen = image.ImageDataGenerator(
    featurewise_center = True,             # Boolean. Set input mean to 0 over the dataset, feature-wise
    samplewise_center = False,              # Boolean. Set each sample mean to 0
    featurewise_std_normalization = True,  # Boolean. Divide inputs by std of the dataset, feature-wise
    samplewise_std_normalization = False,   # Boolean. Divide each input by its std
    zca_whitening = True,                  # Boolean. Apply ZCA whitening
    rotation_range = 15,                    # Int. Degree range for random rotations
    width_shift_range = 0.15,                # Float. Range for random horizontal shifts
    height_shift_range = 0.15,               # Float. Range for random vertical shifts
    shear_range = 0.12,                      # Float. Shear Intensity
    zoom_range = 0.12,                       # Float. Range for random zoom
    fill_mode = 'nearest',                  # Points outside the boundaries of the input are filled according to the default nearest state
    horizontal_flip = False,                 # Boolean. Randomly flip inputs horizontally
    vertical_flip = False)                  # Boolean. Randomly flip inputs vertically

tr_generator = tr_datagen.flow_from_directory(
    './Heiro_train_handwriting/',  # this is where the training data is
    target_size=image_shape,  # all images should be resized to 32x32
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical')

# validation data generation and preprocessing
val_datagen = image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rescale=None,
                                       rotation_range=15,
                                       width_shift_range=0.15,
                                       height_shift_range=0.15,
                                       shear_range=0.12,
                                       zoom_range=0.12,
                                       fill_mode='nearest',
                                       horizontal_flip=False,
                                       zca_whitening=True)
# this is a similar generator, for validation data
val_generator = val_datagen.flow_from_directory(
    './Heiro_val_handwriting/',
    target_size=image_shape,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb')

# load pre-trained Xception CNN model on imagenet weights
base_model = Xception(weights='imagenet', pooling = 'avg', include_top=False)

#for i in base_model.layers[:20]: # Freezing the first 40 layers
#    i.trainable = False
#    print(i.name)

# add a global spatial average pooling layer
x = base_model.output

# add the classification layer
predictions1 = Dense(40, activation='softmax')(x)

#create the model
model = Model(input=base_model.input, output=predictions1)

# #load previous weights (after first trainings)
model.load_weights('./theweights_xception_handwriting.hdf5')

# the_weights_xceptionhandwriting : 0.278 best val_loss

# Stochastic Gradient Descent optimizer.
sgd = SGD(lr=0.005, momentum=0.5, decay=0.0001, nesterov=False)

model.compile(loss = 'categorical_crossentropy', optimizer = sgd,  metrics = ['accuracy'])
model.summary()

#reading current time to differentiate files in logs and submissions.
current = time.strftime("%c")

# Saving the weights and creating a logs directory for Tensorboard
checkpointer = ModelCheckpoint(filepath = "theweights_xception_handwriting2.hdf5", verbose = 1, save_best_only = True, monitor = 'val_loss')
tb = TensorBoard(log_dir='./logs_hand_xcept/'+ current, histogram_freq=0, write_graph=True, write_images=False)

# Using reduce Learning rate on plateau to try to prevent saturation and overfitting by reducing the learning rate.
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.85, patience = 6, min_lr =1e-5, verbose=1)

# Train model
model.fit_generator(
        tr_generator,
        steps_per_epoch=1200/batch_size, # amount of data we want to train on
        epochs=epoch_count,
        validation_data=val_generator,
        validation_steps=300/batch_size, # amount of data we want to validate on
        callbacks = [tb, checkpointer, reduce_lr])
# Train model
# model.fit_generator(
#     tr_generator,
#     samples_per_epoch=1000,  # amount of data we want to train on
#     nb_epoch=epoch_count,
#     validation_data=val_generator,
#     nb_val_samples=400,  # amount of data we want to validate on
#     callbacks=[tb, checkpointer])

# # Using the testing file to create a submission
#
# # Load most recent weights generated
# model.load_weights('./theweights.hdf5')
# model.compile(loss = 'categorical_crossentropy', optimizer = sgd,  metrics = ['accuracy'])
#
# # copy to my folder to keep track of all weights
# cmd = 'cp theweights.hdf5 theweights/my_weights_'+ current +'.hdf5'
# os.system(cmd)
#
# l=[None]*200
#
# for key,value in tr_generator.class_indices.items():
#     l[value]=key
#
# f = open('mysub'+current, 'w')
#
# f.write('Id,Prediction\n')
#
# for i in range(0, 10000):
#     imgn = 'test_%d' % (i,)
#     imgn += '.JPEG'
#     img_p = './test/' + imgn
#     img = image.load_img(img_p, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#
#     # Predict for test image
#     preds = model.predict(x)
#
#     # write the result line using highest probability from the classifier
#     f.write(imgn + ',' + l[np.argmax(preds)] + '\n')
#
#
# f.close()
