from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import json
from datetime import datetime

app = Flask(__name__)
app.config['SAVE_FOLDER'] = 'static/dataset'

@app.route('/')
def index():
    return render_template('index.html', message=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        captions = request.form['captions']
        createdby = request.form['createdby']
        # createdin: type datetime
        notes = request.form['notes']

        image_path = os.path.join(app.config['SAVE_FOLDER'], image.filename)
        image.save(image_path)

        data = {
            'name': image.filename,
            'captions': captions,
            'createdby': createdby,
            'createdin': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'notes': notes,
            'image_path': image_path
        }
        save_to_json(data)
        return redirect(url_for('success'))
    return render_template('index.html')

@app.route('/success')
def success():
    return render_template('success.html')
def save_to_json(data):
    json_file = 'data.json'
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            existing_data = json.load(f)
        existing_data.append(data)
        with open(json_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
    else:
        with open(json_file, 'w') as f:
            json.dump([data], f, indent=2)

if __name__ == '__main__':
    app.run(debug=True)
