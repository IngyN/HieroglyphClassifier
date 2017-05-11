import keras
import numpy as np
import time

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau     # Reference: https://keras.io/callbacks/
from keras.applications import *
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Activation, Flatten, GlobalAveragePooling2D
from keras.constraints import maxnorm
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.optimizers import Adam, SGD

current = time.strftime("%c")

def load_dataset():
    validation_data_dir = './Heiroglyphs/'

    classes = []
    for subdir in sorted(os.listdir(validation_data_dir)):
        if os.path.isdir(os.path.join(validation_data_dir, subdir)):
            classes.append(subdir)

    class_indices = dict(zip(classes, range(len(classes))))

    X_val = []

    # Extracting validation dat
    i = 0
    y_val = []
    for subdir in classes:
        subpath = os.path.join(validation_data_dir, subdir)
        for fname in sorted(os.listdir(subpath)):
            y_val.append(class_indices[subdir])

            # Load image as numpy array and append it to X_val
            img = load_img(os.path.join(subpath, fname), target_size=(img_width, img_height))
            x = img_to_array(img)
            X_val.append(x)

            i += 1

    Y_val = np_utils.to_categorical(y_val)
    X_val = np.asarray(X_val, dtype='float32')
    return classes, X_val, Y_val

#LOADING DATASET 
classes, X_train, Y_train = load_dataset()

#Labels to categorical 
Y_train = np_utils.to_categorical(y_train, nb_classes)

############## Creating Model ################

# initialize the model
model = Sequential()

# first set of CONV => RELU => POOL
model.add(Convolution2D(20, 5, 5, border_mode="same",
        input_shape=(depth, height, width)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# second set of CONV => RELU => POOL
model.add(Convolution2D(50, 5, 5, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# set of FC => RELU layers
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

# softmax classifier
model.add(Dense(classes))
model.add(Activation("softmax"))

model.load_weights("lenet_weights.hdf5")

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
# for layer in base_model.layers:
#      layer.trainable = False

sgd = SGD(lr=0.01, momentum=0.6, decay=0.000001, nesterov=True)

model.compile(loss = 'categorical_crossentropy', optimizer = sgd,  metrics = ['accuracy'])
model.summary()


# Training Model

checkpointer = ModelCheckpoint(filepath = "heiro.hdf5", verbose = 1, save_best_only = True, monitor = 'val_loss')   
tb = TensorBoard(log_dir='./heiro_logs/'+ current, histogram_freq=0, write_graph=True, write_images=False)

#reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 5, min_lr = 1e-6)

history = model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1, callbacks=[ checkpointer, tb] , validation_split=0.10)



