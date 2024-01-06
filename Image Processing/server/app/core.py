from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras.preprocessing.image import image_utils
from PIL import Image
import numpy as np

import tensorflow as tf


def preprocess_image(image_path):
    image = Image.open(image_path)
    img = image.resize((224, 224))
    return img

def make_prediction(model, processed_image):
    img = image_utils.load_img(processed_image,
                         target_size=(224, 224))
    print(processed_image)
    x = image_utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    predictions = model.predict(img_data)
    print(predictions)
    class_labels = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']
    score = tf.nn.softmax(predictions[0])
    print(f"{class_labels[tf.argmax(score)]}")
    return f"{class_labels[tf.argmax(score)]}"