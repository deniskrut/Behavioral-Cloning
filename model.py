from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
import csv
import numpy as np


# Based on this example https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(160,320,3,)))
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

with open('model.json', 'w') as jfile:
    jfile.write(model.to_json())

base_path = "/Users/denyskrut/Documents/CarND-behavioral-cloning/"
base_path_len = len(base_path)

def load_image(img_path):
    relative_path = "./" + img_path[base_path_len:]
    img = image.load_img(relative_path)
    x = image.img_to_array(img)
    return preprocess_input(x)

steering_angles = []
center_image_paths = []
left_images_paths = []
right_images_paths = []

with open('training/driving_log.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        steering_angle = float(row[3])
        steering_angles.append(steering_angle)
        
        center_image_paths.append(row[0])
        left_images_paths.append(row[1])
        right_images_paths.append(row[2])

def train_model(paths, steering_angles, epoch_count):
    images = np.asarray([load_image(path) for path in paths])

    model.fit(images, steering_angles, nb_epoch=epoch_count,
              verbose=1, validation_split=0.1)

steering_angles = np.array(steering_angles)

steering_coefficient = 0.25

steering_angles_left = steering_angles - steering_coefficient
steering_angles_right = steering_angles + steering_coefficient

model.compile(loss='mse',
              optimizer=Adam())

nb_epoch = 3

train_model(center_image_paths, steering_angles, nb_epoch)
train_model(left_images_paths, steering_angles_left, nb_epoch)
train_model(right_images_paths, steering_angles_right, nb_epoch)

model.save_weights('model.h5')
