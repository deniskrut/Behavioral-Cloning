from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import csv
import numpy as np
from PIL import ImageOps
from sklearn.model_selection import train_test_split

# Model based on this example https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(32,32,3,)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

# Storing model in file
with open('model.json', 'w') as jfile:
    jfile.write(model.to_json())

# Path substitution for runing in machine different then training
base_path = "/Users/denyskrut/Documents/CarND-behavioral-cloning/"
base_path_len = len(base_path)

# Method to load image form file. Accepts "mirror" parameter to get more data.
def load_image(img_path, mirror = False):
    relative_path = "./" + img_path[base_path_len:]
    img = image.load_img(relative_path, target_size=(32,32))
    if (mirror):
        img = ImageOps.mirror(img)
    x = image.img_to_array(img)
    return preprocess_input(x)

# Arrays to store the data from file
steering_angles = []
center_image_paths = []
left_images_paths = []
right_images_paths = []

# Read data from files to the arrays
with open('training/driving_log.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        steering_angle = float(row[3])
        steering_angles.append(steering_angle)
        
        center_image_paths.append(row[0])
        left_images_paths.append(row[1])
        right_images_paths.append(row[2])

# Method to load images from paths
def load_images(paths, mirror = False):
    return np.array([load_image(path, mirror) for path in paths])

# Steering coefficient to add or substract from left and right training images
steering_coefficient = 0.1

# Generate steering angles - center, left, right and center mirrored
steering_angles = np.array(steering_angles)

steering_angles_mirror = steering_angles * -1.
steering_angles_left = steering_angles - steering_coefficient
steering_angles_right = steering_angles + steering_coefficient

steering_angles = np.concatenate((steering_angles, steering_angles_mirror))
steering_angles = np.concatenate((steering_angles, steering_angles_left))
steering_angles = np.concatenate((steering_angles, steering_angles_right))

# Load center, left, right and center mirrored images
images = load_images(center_image_paths)
images = np.concatenate((images, load_images(center_image_paths, mirror = True)))
images = np.concatenate((images, load_images(left_images_paths)))
images = np.concatenate((images, load_images(right_images_paths)))

# Split data on training and validation
images_train, images_val, steering_angles_train, steering_angles_val = train_test_split(images, steering_angles, test_size=0.1, random_state=424242)

# Compile model
model.compile(loss='mse',
              optimizer=Adam())

# Create a generator that would shift weights, heights and channels channels, and also zoom.
image_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, channel_shift_range=0.1, fill_mode='nearest')

# Define number of epochs
nb_epoch = 3

# Train the model using generator
model.fit_generator(image_generator.flow(images_train, steering_angles_train), samples_per_epoch=len(steering_angles), nb_epoch=nb_epoch, verbose=1, validation_data=image_generator.flow(images_val, steering_angles_val), nb_val_samples=len(steering_angles_val))

# Save weights
model.save_weights('model.h5')
