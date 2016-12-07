from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.constraints import maxnorm
from keras.regularizers import l2, activity_l2
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
#W_constraint = maxnorm(2), W_regularizer=l2(0.1), activity_regularizer=activity_l2(0.1)
#x = Dropout(0.05)(x)
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

nb_epoch = 5

train_model(center_image_paths, steering_angles, nb_epoch)
train_model(left_images_paths, steering_angles_left, nb_epoch)
train_model(right_images_paths, steering_angles_right, nb_epoch)

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

train_model(center_image_paths, steering_angles, nb_epoch)
train_model(left_images_paths, steering_angles_left, nb_epoch)
train_model(right_images_paths, steering_angles_right, nb_epoch)

model.save_weights('model.h5')
