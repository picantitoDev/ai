from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cargar modelo
modelo = load_model("modelov6.h5")

# Diccionario de clases
clases = {0: "No Parkinson", 1: "Indicios de Parkinson"}

# Función de predicción
def predecir_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = modelo.predict(img)[0][0]
    resultado = clases[1 if pred > 0.5 else 0]
    prob = round(pred * 100, 2)
    return resultado, prob

# Rutas
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'imagen' not in request.files:
            return redirect(request.url)
        file = request.files['imagen']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            ruta = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(ruta)
            resultado, prob = predecir_imagen(ruta)
            return render_template('index.html', resultado=resultado, prob=prob, imagen=filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
