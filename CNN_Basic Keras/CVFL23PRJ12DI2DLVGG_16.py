import keras
from keras import layers

model = keras.models.Sequential()

model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block1_conv1',
           input_shape=(224, 224, 3)))
model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block1_conv2'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_maxpool'))

model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block2_conv1'))
model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block2_conv2'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_maxpool'))

model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block3_conv1'))
model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block3_conv2'))
model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block3_conv3'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_maxpool'))

model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block4_conv1'))
model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block4_conv2'))
model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block4_conv3'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_maxpool'))

model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block5_conv1'))
model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block5_conv2'))
model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block5_conv3'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_maxpool'))

model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()