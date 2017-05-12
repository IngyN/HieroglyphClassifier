from __future__ import print_function
import numpy as np
np.random.seed(1234)  # for reproducibility
import os 
from keras.datasets import cifar10
from keras.models import Sequential, load_model, model_from_json
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
##from keras.regularizers.WeightRegularizer import W_regularizer
from keras.regularizers import l1, l2
from keras.preprocessing.image import ImageDataGenerator,img_to_array,  load_img 
from keras.preprocessing import image

def load_dataset():
    validation_data_dir = './extra/'
    
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
            img = load_img(os.path.join(subpath, fname), target_size=(32, 32))
            x = img_to_array(img)
            X_val.append(x)
            
            i += 1

    Y_val = np_utils.to_categorical(y_val)
    X_val = np.asarray(X_val, dtype='float32')
    return classes, X_val, Y_val


testgen = ImageDataGenerator(featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False)


test_gen = testgen.flow_from_directory('./extra/',  # this is the target directory
                                        target_size=(32, 32),
                                        batch_size=32,
                                        class_mode='categorical',
                                        color_mode = 'grayscale')##Model##

f = open('augmented_model_architecture.json', 'r')
model = model_from_json(f.read())
f.close()

# Load weights
model.load_weights('theweights.hdf5', by_name=True)
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer =  SGD, metrics =[ 'accuracy'] )

score = model.evaluate_generator(test_gen, val_samples = 240 )
print('Test score:', score[0])
print('Test accuracy:', score[1])
pred = model.predict_generator(test_gen, 240)

print( pred)









