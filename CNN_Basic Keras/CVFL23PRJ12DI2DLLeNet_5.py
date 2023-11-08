import keras
from keras import layers

model = keras.models.Sequential()
model.add(layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.AveragePooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(layers.AveragePooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()