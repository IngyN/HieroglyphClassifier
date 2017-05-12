from __future__ import print_function
import numpy as np
np.random.seed(1234)  # for reproducibility

from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
##from keras.regularizers.WeightRegularizer import W_regularizer
from keras.regularizers import l1, l2
from keras.preprocessing.image import ImageDataGenerator

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
            img = load_img(os.path.join(subpath, fname), target_size=(img_width, img_height))
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
                                        target_size=(28,28),
                                        batch_size=batch_size,
                                        class_mode='categorical',
                                        color_mode = 'grayscale')##Model##

f = open('augmented_model_architecture.json', 'r')
model = model_from_json(f.read())
f.close()

# Load weights
model.load_weights('theweights.hdf5', by_name=True)
model.summary()


classes, X, Y = load_dataset()

l=[None]*40

for key,value in test_gen.class_indices.items():
    l[value]=key

f = open('testing.csv', 'w')
f.write('Id,Prediction\n')

for i, (img, target ) in enumerate ( zip ( X, Y )) :
     y= model.predict(img )
     f.write(img_name + ',' + l[np.argmax(preds)] + '\n')

f.close()


#for i in range(0, 10000):
#    img_name = 'test_%d' % (i,)
#    img_name += '.JPEG'
#    img_path = './test_images/' + img_name
#    img = image.load_img(img_path, target_size=(299, 299))
#    x = image.img_to_array(img)
#    x = np.expand_dims(x, axis=0)
#    # x = preprocess_input(x)
#    preds = model.predict(x)
#    f.write(img_name + ',' + l[np.argmax(preds)] + '\n')
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3)[0])
# print()


#score = model.evaluate_generator(test_gen, val_samples = y_test.shape[0] )
#print('Test score:', score[0])
#print('Test accuracy:', score[1])
#









