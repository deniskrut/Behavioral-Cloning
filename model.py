from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
import csv
import numpy as np

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

base_path = "/Users/denyskrut/Documents/CarND-behavioral-cloning/"
base_path_len = len(base_path)

image_size = (320, 160)
image_chanels = 3

def load_image(img_path):
    relative_path = "./" + img_path[base_path_len:]
    img = image.load_img(relative_path)
    x = image.img_to_array(img)
    return preprocess_input(x)

steering_angles = []
center_images = []
left_images = []
right_images = []

with open('training/driving_log.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        steering_angle = float(row[3])
        steering_angles.append(steering_angle)
        
        center_image = load_image(row[0])
        left_image = load_image(row[1])
        right_image = load_image(row[2])
        
        center_images.append(center_image)
        #left_images.append(left_image)
        #right_images.append(right_image)

steering_angles = np.array(steering_angles)
center_images = np.array(center_images)
#left_images = np.asarray(left_images)
#right_images = np.asarray(right_images)

steering_coefficient = 0.25

steering_angles_left = steering_angles - steering_coefficient
steering_angles_right = steering_angles + steering_coefficient

model.compile(loss='mse',
              optimizer=Adam())

nb_epoch = 5
    
model.fit(center_images, steering_angles, nb_epoch=nb_epoch,
          verbose=1, validation_split=0.1)
#model.fit(left_images, steering_angles_left, nb_epoch=nb_epoch,
#      verbose=1, validation_split=0.1)
#model.fit(right_images, steering_angles_right, nb_epoch=nb_epoch,
#          verbose=1, validation_split=0.1)

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
model.compile(loss='mse',
              optimizer=Adam())

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit(center_images, steering_angles, nb_epoch=nb_epoch,
          verbose=1, validation_split=0.1)
#model.fit(left_images, steering_angles_left, nb_epoch=nb_epoch,
#          verbose=1, validation_split=0.1)
#model.fit(right_images, steering_angles_right, nb_epoch=nb_epoch,
#          verbose=1, validation_split=0.1)

model.save_weights('model.h5')
