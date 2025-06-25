import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your trained model
model = tf.keras.models.load_model('model/parkinson_disease_model.h5')

# Helper to check allowed image extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image (customize this as needed)
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # Change this to your model's expected size
    img_array = np.array(img) / 255.0  # Normalize if your model expects it
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess and predict
        image = preprocess_image(filepath)
        prediction = model.predict(image)

        # Process output (customize depending on model output)
        predicted_class = np.argmax(prediction, axis=1)[0]  # if categorical
        confidence = float(np.max(prediction))

        return render_template('index.html',
                               prediction=predicted_class,
                               confidence=round(confidence * 100, 2))

    else:
        flash('Invalid file type. Please upload a PNG or JPG image.')
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
