import tensorflow as tf
from tensorflow import keras

# load the dataset
(train_img,train_label),(test_img,test_label) = keras.datasets.mnist.load_data()
train_img = train_img.reshape([-1, 28, 28, 1])
test_img = test_img.reshape([-1, 28, 28, 1])
train_img = train_img/255.0
test_img = test_img/255.0
train_label = keras.utils.to_categorical(train_label)
test_label = keras.utils.to_categorical(test_label)

# define the model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (5, 5), padding="same", input_shape=[28, 28, 1]),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(64, (5, 5), padding="same"),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_img,train_label, validation_data=(test_img,test_label), epochs=10)
test_loss,test_acc = model.evaluate(test_img, test_label)
print('Test accuracy:', test_acc)

print(test_img.shape)

import matplotlib.pyplot as plt
plt.imshow(test_img[0,:,:,:],cmap='gray')
plt.show()

pred = model.predict(test_img[0:100,:,:,:])
model.save('digit_model.keras')

from keras.models import load_model
model = load_model("digit_model.keras")
pred = model.predict(test_img[:5,:,:,:])

print(pred)

i = 0
for el in pred:
  #max_ele = max(el)
  #
  #
  max_index = el.argmax()
  plt.imshow(test_img[i,:,:,:])
  plt.title(str(max_index))
  plt.show()
  i += 1