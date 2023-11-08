import keras
from keras import layers

model = keras.models.Sequential()
model.add(layers.Input(shape=(28, 28)))
model.add(layers.SimpleRNN(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()