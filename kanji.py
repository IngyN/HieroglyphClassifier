import sys
import numpy as np
from keras.models import model_from_json
import h5py

batch_size = 32 # batch size
epoch_count = 10 # Number of epochs to train
image_shape = (32, 32)

def load_dataset():
    validation_data_dir = './Heiroglyphs/'

    classes = []
    for subdir in sorted(os.listdir(validation_data_dir)):
        if os.path.isdir(os.path.join(validation_data_dir, subdir)):
            classes.append(subdir)

    class_indices = dict(zip(classes, range(len(classes))))

    X_val = []

    # Extracting validation data
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


# Load and augment training data
tr_datagen = image.ImageDataGenerator(
    featurewise_center = True,             # Boolean. Set input mean to 0 over the dataset, feature-wise
    samplewise_center = False,              # Boolean. Set each sample mean to 0
    featurewise_std_normalization = True,  # Boolean. Divide inputs by std of the dataset, feature-wise
    samplewise_std_normalization = False,   # Boolean. Divide each input by its std
    zca_whitening = False,                  # Boolean. Apply ZCA whitening
    rotation_range = 10,                    # Int. Degree range for random rotations
    width_shift_range = 0.10,                # Float. Range for random horizontal shifts
    height_shift_range = 0.10,               # Float. Range for random vertical shifts
    shear_range = 0.25,                      # Float. Shear Intensity
    zoom_range = 0.25,                       # Float. Range for random zoom
    fill_mode = 'nearest',                  # Points outside the boundaries of the input are filled according to the default nearest state
    horizontal_flip = True,                 # Boolean. Randomly flip inputs horizontally
    vertical_flip = False)                  # Boolean. Randomly flip inputs vertically

tr_generator = tr_datagen.flow_from_directory(
        './data/',  # this is where the training data is
        target_size=image_shape,  # all images should be resized to 32x32
        batch_size=batch_size,
        class_mode='categorical')

# validation data generation and preprocessing
val_datagen = image.ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True,rescale = None)

# Load architecture
f = open('model_architecture.json', 'r')
model = model_from_json(f.read())
f.close()

# Load weights
model.load_weights('model_weights.h5')

checkpointer = ModelCheckpoint(filepath = "theweights.hdf5", verbose = 1, save_best_only = True, monitor = 'val_loss')
tb = TensorBoard(log_dir='./logsKanji/'+ current, histogram_freq=0, write_graph=True, write_images=False)

# Using reduce Learning rate on plateau to try to prevent saturation and overfitting by reducing the learning rate.
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 4, min_lr =1e-6)

# Train model
model.fit_generator(
        tr_generator,
        steps_per_epoch=50, # amount of data we want to train on
        epochs=epoch_count,
        validation_data=val_generator,
        validation_steps=batch_size, # amount of data we want to validate on
        callbacks = [tb, checkpointer])

        
