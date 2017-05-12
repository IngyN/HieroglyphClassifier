import sys
import os
import keras
import time
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau  # Reference: https://keras.io/callbacks/
import h5py
# import PIL

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

# Load architecture
f = open('augmented_model_architecture.json', 'r')
model = model_from_json(f.read())
f.close()
# Stochastic Gradient Descent optimizer.
sgd = SGD(lr=0.00002, momentum=0.7, decay=0.0001, nesterov=True)
current = time.strftime("%c")


# Using the testing file to create a submission

# Load most recent weights generated
model.load_weights('./theweights.hdf5')
model.compile(loss = 'categorical_crossentropy', optimizer = sgd,  metrics = ['accuracy'])

# # copy to my folder to keep track of all weights
# cmd = 'cp theweights.hdf5 theweights/my_weights_'+ current[0:5] +'.hdf5'
# os.system(cmd)

l=[None]*40

for key,value in tr_generator.class_indices.items():
    l[value]=key

f = open('mysub'+current[0:5], 'w')

f.write('Id,Prediction\n')

for i in range(0, 13):
    imgn = 'test_%d' % (i,)
    imgn += '.jpg'
    img_p = './test/' + imgn
    img = image.load_img(img_p, target_size=(32, 32))
    img = img.convert('L')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Predict for test image
    preds = model.predict(x)

    # write the result line using highest probability from the classifier
    f.write(imgn + ',' + l[np.argmax(preds)] + '\n')


f.close()