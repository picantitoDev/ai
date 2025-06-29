import os
import numpy as np
import tensorflow as tf
import base64
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from datetime import datetime

# Configuración
UPLOAD_FOLDER = os.path.join('static', 'preview')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

# Crear carpeta si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cargar modelo
model_path = os.path.join(os.path.dirname(__file__), 'model', 'parkinson_disease_detection.h5')
model = tf.keras.models.load_model(model_path)

# Helper function to convert image to base64
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Extensiones permitidas
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# FIXED: Separate preprocess_image function
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Escala de grises
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # canal
    img_array = np.expand_dims(img_array, axis=0)   # batch
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Create unique filename to avoid conflicts
        timestamp = str(int(datetime.now().timestamp()))
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print("Imagen guardada en:", filepath)
        print("Archivo existe:", os.path.exists(filepath))
        
        # Store only filename and timestamp in session (no base64 to avoid session size limits)
        session['filename'] = filename
        session['timestamp'] = datetime.now().timestamp()
        
        return render_template('index.html',
                               uploaded=True,
                               filename=filename,
                               image_base64=None,  # Don't pass base64 to avoid session bloat
                               timestamp=session['timestamp'])
    else:
        flash('Invalid file type. Please upload a PNG or JPG image.')
        return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    filename = session.get('filename')
    if not filename:
        flash('No image uploaded. Please upload an image first.')
        return redirect(url_for('index'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        flash('Image file not found. Please upload again.')
        return redirect(url_for('index'))
    
    try:
        # Predecir
        image = preprocess_image(filepath)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        label = "Parkinson" if predicted_class == 1 else "Healthy"
        
        return render_template('index.html',
                               uploaded=True,
                               filename=filename,
                               image_base64=None,  # Don't use base64 to avoid session size issues
                               timestamp=session.get('timestamp'),
                               prediction=label,
                               confidence=round(confidence * 100, 2))
    except Exception as e:
        print(f"Error during prediction: {e}")
        flash('Error processing image. Please try again.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)