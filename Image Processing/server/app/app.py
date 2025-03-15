from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from core import make_prediction

app = Flask(__name__)
app.config['SAVE_FOLDER'] = 'static/download'
model = tf.keras.models.load_model('models/YogaNet_Model_3_BaseCNN.h5')

# Initialize Firebase Admin SDK
cred = credentials.Certificate('firebase_config.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

def format_datetime(value):
    dt = datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%f')
    return dt.strftime('%Y-%m-%d %H:%M:%S')

app.jinja_env.filters['datetime'] = format_datetime
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    image_path = os.path.join(app.config['SAVE_FOLDER'], image.filename)
    image_path = image_path.replace("\\", "/")
    image.save(image_path)

    prediction = make_prediction(model, image_path)

    # Save prediction to Firebase
    doc_ref = db.collection('predictions').document()
    doc_ref.set({
        'timestamp': datetime.now().isoformat(),
        'image_name': image.filename,
        'prediction': prediction
    })

    os.remove(image_path)
    return render_template('predict.html', image_filename='download/' + image.filename, prediction=prediction)


@app.route('/history')
def history():
    # Get prediction history from Firebase
    predictions_ref = db.collection('predictions').order_by('timestamp', direction=firestore.Query.DESCENDING)
    docs = predictions_ref.stream()
    predictions = [{'timestamp': doc.to_dict()['timestamp'], 'image_name': doc.to_dict()['image_name'],
                    'prediction': doc.to_dict()['prediction']} for doc in docs]

    return render_template('history.html', predictions=predictions)


if __name__ == '__main__':
    app.run(debug=True)
