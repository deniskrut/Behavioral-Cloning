from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import csv
from scipy.misc import imread
import numpy as np

model = Sequential()
model.add(Flatten(input_shape=(160,320,3,)))
model.add(BatchNormalization())
model.add(Dense(128))
model.add(Dense(1))

with open('model.json', 'w') as jfile:
    jfile.write(model.to_json())

steering_angles = []
center_images = []

with open('training/driving_log.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        steering_angle = float(row[3])
        steering_angles.append(steering_angle)
        
        center_file = row[0]
        center_image = imread(center_file)
        center_images.append(center_image)

steering_angles = np.array(steering_angles)
center_images = np.array(center_images)

model.compile(loss='mse',
              optimizer=Adam(),
              metrics=['accuracy'])

nb_epoch = 50
    
model.fit(center_images, steering_angles, nb_epoch=nb_epoch,
          verbose=1)

model.save_weights('model.h5')
