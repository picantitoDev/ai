# ===== app.py =====
from flask import Flask, render_template, request, flash, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename
import logging

# Configuración de la aplicación Flask
app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_aqui'  # Cambia por una clave segura en producción

# Configuración de directorios
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Límite de 16MB

# Crear directorio de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configurar logging para debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales para el modelo
modelo = None
IMG_SIZE = (224, 224)  # Tamaño esperado por el modelo
CLASES = ['Healthy', 'Parkinson']  # Basado en tu código original

def cargar_modelo():
    """
    Carga el modelo CNN entrenado desde el archivo .h5
    """
    global modelo
    try:
        modelo_path = 'src/model/modelov3.keras'  
        if os.path.exists(modelo_path):
            modelo = load_model(modelo_path)
            logger.info(f"Modelo cargado exitosamente desde {modelo_path}")
            logger.info(f"Forma de entrada esperada: {modelo.input_shape}")
        else:
            logger.error(f"No se encontró el archivo del modelo: {modelo_path}")
            return False
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        return False
    return True

def archivo_permitido(filename):
    """
    Verifica si el archivo tiene una extensión permitida
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocesar_imagen(imagen_path):
    """
    Preprocesa la imagen para que sea compatible con el modelo:
    - Redimensiona a (128, 128)
    - Convierte a RGB
    - Normaliza valores entre 0 y 1
    - Agrega dimensión de batch
    """
    try:
        # Abrir y convertir imagen
        img = Image.open(imagen_path)
        
        # Convertir a RGB (asegurar 3 canales)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar a tamaño esperado por el modelo
        img = img.resize(IMG_SIZE)
        
        # Convertir a array numpy
        img_array = np.array(img)
        
        # Normalizar valores entre 0 y 1
        img_array = img_array.astype(np.float32) / 255.0
        
        # Agregar dimensión de batch (1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Imagen preprocesada - Shape: {img_array.shape}")
        return img_array
        
    except Exception as e:
        logger.error(f"Error al preprocesar imagen: {str(e)}")
        return None

def clasificar_imagen(imagen_path):
    """
    Realiza la clasificación de la imagen usando el modelo cargado
    """
    global modelo
    
    if modelo is None:
        return None, "Modelo no cargado"
    
    try:
        # Preprocesar imagen
        img_array = preprocesar_imagen(imagen_path)
        if img_array is None:
            return None, "Error al preprocesar la imagen"
        
        # Realizar predicción
        prediccion = modelo.predict(img_array, verbose=0)
        
        # Interpretar resultado (asumiendo clasificación binaria con sigmoid)
        if len(prediccion.shape) > 1 and prediccion.shape[1] == 1:
            # Salida sigmoid (0-1)
            confianza = float(prediccion[0][0])
            if confianza > 0.5:
                clase_predicha = CLASES[1]  # Parkinson
                porcentaje_confianza = confianza * 100
            else:
                clase_predicha = CLASES[0]  # Healthy
                porcentaje_confianza = (1 - confianza) * 100
        else:
            # Salida softmax o múltiples clases
            clase_idx = np.argmax(prediccion[0])
            clase_predicha = CLASES[clase_idx]
            porcentaje_confianza = float(np.max(prediccion[0])) * 100
        
        resultado = {
            'clase': clase_predicha,
            'confianza': round(porcentaje_confianza, 2),
            'prediccion_raw': float(prediccion[0][0]) if len(prediccion.shape) > 1 and prediccion.shape[1] == 1 else prediccion[0].tolist()
        }
        
        logger.info(f"Clasificación exitosa: {resultado}")
        return resultado, None
        
    except Exception as e:
        error_msg = f"Error en la clasificación: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Ruta principal que maneja tanto la visualización como el procesamiento
    """
    if request.method == 'POST':
        # Verificar si se subió un archivo
        if 'archivo' not in request.files:
            flash('No se seleccionó ningún archivo', 'error')
            return redirect(request.url)
        
        archivo = request.files['archivo']
        
        # Verificar si se seleccionó un archivo
        if archivo.filename == '':
            flash('No se seleccionó ningún archivo', 'error')
            return redirect(request.url)
        
        # Procesar archivo si es válido
        if archivo and archivo_permitido(archivo.filename):
            try:
                # Guardar archivo de forma segura
                filename = secure_filename(archivo.filename)
                timestamp = str(int(np.random.rand() * 1000000))  # Evitar colisiones de nombres
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                archivo.save(filepath)
                
                # Realizar clasificación
                resultado, error = clasificar_imagen(filepath)
                
                if error:
                    flash(f'Error al procesar la imagen: {error}', 'error')
                    return render_template('index.html')
                
                # Mostrar resultado
                return render_template('index.html', 
                                     resultado=resultado, 
                                     imagen_url=f'uploads/{filename}')
                                     
            except Exception as e:
                flash(f'Error al procesar el archivo: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Tipo de archivo no permitido. Use: PNG, JPG, JPEG, GIF, BMP', 'error')
            return redirect(request.url)
    
    # GET request - mostrar formulario
    return render_template('index.html')

@app.errorhandler(413)
def archivo_muy_grande(error):
    """Manejo de archivos demasiado grandes"""
    flash('El archivo es demasiado grande. Máximo 16MB permitido.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Cargar modelo al iniciar la aplicación
    if not cargar_modelo():
        print("❌ Error: No se pudo cargar el modelo. Verifica que el archivo 'modelo_cnn.h5' existe.")
        print("   El archivo debe estar en la raíz del proyecto.")
        exit(1)
    
    print("✅ Modelo cargado exitosamente")
    print("🚀 Iniciando aplicación Flask...")
    print("   Accede a: http://localhost:5000")
    
    # Ejecutar aplicación
    app.run(debug=True, host='0.0.0.0', port=5000)