import keras
from keras import layers

model = keras.models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(1, 1), padding='valid', activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPool2D(2, 2))

model.add(layers.Conv2D(32, (5, 5), strides=(1, 1), padding='valid', activation='relu'))
model.add(layers.MaxPool2D(2, 2))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()