from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam
import csv
from scipy.misc import imread
import numpy as np
from keras.applications.inception_v3 import InceptionV3

#using example from Keras documentation https://keras.io/applications/

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer
predictions = Dense(1)(x)

model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

with open('model.json', 'w') as jfile:
    jfile.write(model.to_json())

steering_angles = []
center_images = []
left_images = []
right_images = []

with open('training/driving_log.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        steering_angle = float(row[3])
        steering_angles.append(steering_angle)
        
        center_file = row[0]
        center_image = imread(center_file)
        center_images.append(center_image)

        left_file = row[1]
        left_image = imread(left_file)
        left_images.append(left_image)

        right_file = row[2]
        right_image = imread(right_file)
        right_images.append(right_image)

steering_angles = np.array(steering_angles)
center_images = np.array(center_images)
left_images = np.array(left_images)
right_images = np.array(right_images)

def process_images(images):
    return (center_images - 128.)/256.

center_images = process_images(center_images)
left_images = process_images(left_images)
right_images = process_images(right_images)

model.compile(loss='mse',
              optimizer=Adam())

nb_epoch = 5
    
model.fit(center_images, steering_angles, nb_epoch=nb_epoch,
          verbose=1)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(loss='mse',
              optimizer=Adam())

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit(center_images, steering_angles, nb_epoch=nb_epoch,
          verbose=1)

model.save_weights('model.h5')
