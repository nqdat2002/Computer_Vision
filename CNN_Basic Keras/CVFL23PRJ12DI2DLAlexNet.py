import keras
from keras import layers

model = keras.models.Sequential()

model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), input_shape=(227, 227, 3), padding='valid'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(layers.Conv2D(256, (5, 5), padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(layers.Conv2D(384, (3, 3), padding='same'))
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(384, (3, 3), padding='same'))
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(256, (3, 3), padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(4096))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(4096))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(1000))
model.add(layers.Activation('softmax'))

model.summary()