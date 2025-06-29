from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Inicializar Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cargar modelo .h5
model = load_model("modelo_prueba.h5")

# Extensiones permitidas
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Función para preprocesar la imagen
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # Asegúrate que esto coincide con la entrada de tu modelo
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Ruta principal
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img_tensor = preprocess_image(filepath)
            pred = model.predict(img_tensor)
            prediction = pred.tolist()
        else:
            prediction = "Formato de archivo no permitido. Solo PNG, JPG, JPEG."

    return render_template('index.html', prediction=prediction)

# Ejecutar la app en el puerto 8080
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
