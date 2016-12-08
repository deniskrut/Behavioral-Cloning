from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
import csv
import numpy as np
from PIL import ImageOps


# Based on this example https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

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

with open('model.json', 'w') as jfile:
    jfile.write(model.to_json())

base_path = "/Users/denyskrut/Documents/CarND-behavioral-cloning/"
base_path_len = len(base_path)

def load_image(img_path, mirror = False):
    relative_path = "./" + img_path[base_path_len:]
    img = image.load_img(relative_path, target_size=(32,32))
    if (mirror):
        img = ImageOps.mirror(img)
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

def load_images(paths, mirror = False):
    return np.array([load_image(path, mirror) for path in paths])

steering_coefficient = 0.25

steering_angles = np.array(steering_angles)

steering_angles_mirror = steering_angles * -1.
steering_angles_left = steering_angles - steering_coefficient
steering_angles_right = steering_angles + steering_coefficient

steering_angles = np.concatenate((steering_angles, steering_angles_mirror))
#steering_angles = np.concatenate((steering_angles, steering_angles_left))
#steering_angles = np.concatenate((steering_angles, steering_angles_right))

images = load_images(center_image_paths)
images = np.concatenate((images, load_images(center_image_paths, mirror = True)))
#images = np.concatenate((images, load_images(left_images_paths)))
#images = np.concatenate((images, load_images(right_images_paths)))

model.compile(loss='mse',
              optimizer=Adam())

nb_epoch = 2

model.fit(images, steering_angles, nb_epoch=nb_epoch,
          verbose=1, validation_split=0.1)

model.save_weights('model.h5')
