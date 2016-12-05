from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

model = Sequential()
model.add(Flatten(input_shape=(160,320,3,)))
model.add(Dense(128))
model.add(Dense(1))

with open('model.json', 'w') as jfile:
    jfile.write(model.to_json())

model.save_weights('model.h5')
