from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import tensorflow as tf
from core import make_prediction

app = Flask(__name__)
app.config['SAVE_FOLDER'] = 'static/download'
model = tf.keras.models.load_model('models/YogaNet_Model_3_BaseCNN.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    # image_path = f'static/download/{image.filename}'
    image_path = os.path.join(app.config['SAVE_FOLDER'], image.filename)
    image_path = image_path.replace("\\", "/")
    image.save(image_path)

    # processed_image = preprocess_image(image_path)
    prediction = make_prediction(model, image_path)
    os.remove(image_path)
    return render_template('predict.html', image_filename='download/'+image.filename, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
