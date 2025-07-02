# 🧠 Detección Temprana del Parkinson mediante el Análisis de Trazos de Espirales y Ondas

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-green.svg)](https://flask.palletsprojects.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

> **Sistema de detección temprana del Parkinson basado en análisis computacional de trazos gráficos mediante técnicas de aprendizaje profundo**

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Características Principales](#-características-principales)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Instalación y Configuración](#-instalación-y-configuración)
- [Uso de la Aplicación](#-uso-de-la-aplicación)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Modelo de Aprendizaje Automático](#-modelo-de-aprendizaje-automático)
- [Desarrollo en Google Colab](#-desarrollo-en-google-colab)
- [Dockerización](#-dockerización)
- [Despliegue](#-despliegue)
- [Contribuciones](#-contribuciones)
- [Consideraciones Médicas](#-consideraciones-médicas)
- [Licencia](#-licencia)
- [Información Académica](#-información-académica)

## 🔬 Descripción del Proyecto

Este proyecto implementa un sistema de **detección temprana del Parkinson** utilizando técnicas avanzadas de **aprendizaje profundo** y **visión computacional**. El sistema analiza trazos de espirales y ondas dibujadas por usuarios para identificar patrones característicos asociados con la enfermedad de Parkinson.

La enfermedad de Parkinson afecta el control motor fino, manifestándose en alteraciones específicas en la escritura y dibujo. Nuestro modelo neuronal ha sido entrenado para detectar estas sutiles variaciones en los patrones de movimiento, proporcionando una herramienta de apoyo para la evaluación médica temprana.

### 🎯 Objetivos

- Desarrollar un sistema no invasivo de detección temprana
- Proporcionar una herramienta de apoyo al diagnóstico médico
- Facilitar el acceso a evaluaciones preliminares mediante una interfaz web
- Contribuir a la investigación en neurología computacional

## ✨ Características Principales

- 🖼️ **Análisis de Imágenes**: Procesamiento de trazos de espirales y ondas
- 🧠 **Red Neuronal Profunda**: Modelo entrenado con arquitectura optimizada
- 📊 **Predicción en Tiempo Real**: Resultados instantáneos con porcentaje de probabilidad
- 🌐 **Interfaz Web Intuitiva**: Aplicación Flask con diseño responsivo
- 🔍 **Preprocesamiento Avanzado**: Normalización y extracción de características
- 📈 **Métricas de Confianza**: Sistema de clasificación por niveles de riesgo
- 🐳 **Containerización**: Despliegue simplificado con Docker
- 📱 **Diseño Responsivo**: Compatible con dispositivos móviles y desktop

## 🛠️ Tecnologías Utilizadas

### Backend y ML
- **Python 3.10**: Lenguaje principal de desarrollo
- **TensorFlow 2.19.0**: Framework de aprendizaje profundo
- **OpenCV**: Procesamiento de imágenes y visión computacional
- **NumPy**: Computación numérica y manejo de arrays
- **Flask**: Framework web para la aplicación

### Frontend
- **HTML5**: Estructura de la interfaz
- **CSS3**: Estilos y diseño responsivo
- **JavaScript**: Interactividad del cliente

### Desarrollo y Despliegue
- **Google Colab**: Entorno de desarrollo y entrenamiento
- **Docker**: Containerización de la aplicación
- **Git**: Control de versiones

## 🚀 Instalación y Configuración

### Requisitos del Sistema

- Python 3.8 o superior
- 4GB RAM mínimo (8GB recomendado)
- Espacio en disco: 2GB
- Conexión a internet para descargar dependencias

### Instalación Local

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/proyecto-parkinson.git
cd proyecto-parkinson
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install --upgrade pip
pip install -r requisitos.txt
```

4. **Crear directorios necesarios**
```bash
mkdir -p static/uploads
```

5. **Ejecutar la aplicación**
```bash
python app.py
```

La aplicación estará disponible en `http://localhost:5000`

### Instalación con Docker

1. **Construir la imagen**
```bash
docker build -t parkinson-detector .
```

2. **Ejecutar el contenedor**
```bash
docker run -p 5000:5000 parkinson-detector
```

## 📖 Uso de la Aplicación

### Interfaz Web

1. **Acceder a la aplicación** en tu navegador
2. **Subir una imagen** de trazo (espiral o onda)
3. **Hacer clic en "Subir y Predecir"**
4. **Revisar los resultados** con interpretación automática

### Tipos de Imágenes Aceptadas

- **Formatos**: JPG, JPEG, PNG, BMP
- **Contenido**: Trazos de espirales y ondas dibujadas a mano
- **Resolución**: Mínimo 224x224 píxeles (se redimensiona automáticamente)
- **Calidad**: Imágenes claras con buen contraste

### Interpretación de Resultados

| Rango de Probabilidad | Clasificación | Descripción |
|----------------------|---------------|-------------|
| 0-30% | ✅ Saludable | Imagen saludable detectada |
| 30-50% | ✅ Posiblemente saludable | Resultado favorable |
| 50-70% | ❗ Riesgo moderado | Se recomienda evaluación médica |
| 70-90% | 🧠 Alta probabilidad | Alta probabilidad de Parkinson |
| 90-100% | ⚠️ Muy alta probabilidad | Muy alta probabilidad de Parkinson |

## 📁 Estructura del Proyecto

```
proyecto-parkinson/
├── 📄 README.md                 # Documentación principal
├── 🐍 app.py             # Aplicación Flask principal
├── 🧠 modelov6.h5              # Modelo entrenado de TensorFlow
├── 📋 requisitos.txt           # Dependencias de Python
├── 🐳 Dockerfile               # Configuración de contenedor
├── 🚫 .gitignore               # Archivos ignorados por Git
├── 📁 plantillas/              # Templates HTML
│   └── 🌐 index.html           # Interfaz principal
└── 📁 static/                  # Archivos estáticos (creado automáticamente)
    └── 📁 uploads/             # Imágenes subidas por usuarios
```

### Componentes Principales

#### `app.py`
Aplicación Flask que maneja:
- Carga y procesamiento de imágenes
- Inferencia del modelo neuronal
- Renderizado de templates
- Gestión de archivos estáticos

#### `modelov6.h5`
Modelo de red neuronal convolucional entrenado que incluye:
- Arquitectura de capas convolucionales
- Pesos optimizados mediante entrenamiento
- Funciones de activación especializadas

#### `plantillas/index.html`
Interfaz de usuario con:
- Formulario de carga de imágenes
- Visualización de resultados
- Diseño responsivo y accesible

## 🧠 Modelo de Aprendizaje Automático

### Arquitectura del Modelo

El modelo `modelov6.h5` implementa una **Red Neuronal Convolucional basada en Transfer Learning** con MobileNetV2 como backbone:

```python
# Arquitectura del modelo
- Base Model: MobileNetV2 preentrenado (ImageNet)  
- Input Shape: 224x224x3 (RGB)
- GlobalAveragePooling2D: Reducción dimensional
- Dense Layer: 128 neuronas con activación ReLU
- Dropout: 50% para regularización
- Output Layer: 1 neurona con activación Sigmoid (clasificación binaria)
- Optimizer: Adam (learning_rate=0.0001)
- Loss Function: Binary Crossentropy
```

### Dataset y Preprocesamiento

**Fuente de Datos**: Kaggle - "Handwritten Parkinson's Disease Augmented Data"

**División del Dataset**:
- **Entrenamiento**: 70% de las imágenes
- **Validación**: 15% de las imágenes  
- **Prueba**: 15% de las imágenes

**Técnicas de Augmentación**:
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalización [0,1]
    rotation_range=15,        # Rotación hasta 15°
    zoom_range=0.1,          # Zoom hasta 10%
    width_shift_range=0.1,   # Desplazamiento horizontal
    height_shift_range=0.1,  # Desplazamiento vertical
    horizontal_flip=True     # Volteo horizontal
)
```

### Preprocesamiento de Imágenes

```python
def predecir_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen)           # Cargar imagen
    img = cv2.resize(img, (224, 224))       # Redimensionar a 224x224
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir BGR a RGB
    img = img / 255.0                       # Normalizar [0,1]
    img = np.expand_dims(img, axis=0)       # Añadir dimensión batch
    return modelo.predict(img)[0][0]        # Predicción
```

### Métricas de Evaluación

El modelo fue evaluado con las siguientes métricas médicas:

- **Accuracy**: Porcentaje de predicciones correctas
- **Precision**: Proporción de verdaderos positivos entre predicciones positivas
- **Sensibilidad (Recall)**: Capacidad de detectar casos positivos reales
- **F1-Score**: Media armónica entre precision y recall
- **AUC-ROC**: Área bajo la curva ROC para evaluar discriminación

### Regularización y Control de Entrenamiento

```python
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6),
    ModelCheckpoint("modelo_transfer_parkinson.h5", save_best_only=True)
]
```

## 🔬 Desarrollo en Google Colab

> **Accede al notebook completo de entrenamiento y desarrollo del modelo directamente en Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18W2KbdZMAz3q5c0Y3wW1g9l0HvNMvv1-?usp=sharing)

El modelo fue desarrollado y entrenado completamente en **Google Colab**, aprovechando:

- **GPU gratuita**: Aceleración del entrenamiento con CUDA
- **Entorno preconfigurado**: TensorFlow 2.x y dependencias ML
- **Almacenamiento en la nube**: Integración con Google Drive
- **Notebooks interactivos**: Documentación y código unificados

### Proceso de Desarrollo Completo

#### **Sección 1: Configuración e Importación de Datos**
```python
# Instalación de dependencias
!pip install kaggle tensorflow pandas numpy matplotlib seaborn scikit-learn opencv-python

# Configuración API Kaggle y descarga del dataset
!kaggle datasets download -d banilkumar20phd7071/handwritten-parkinsons-disease-augmented-data
```

#### **Sección 2: Análisis Exploratorio de Datos (EDA)**
- Visualización de distribución de clases (Healthy vs Parkinson)
- Análisis de dimensiones de imágenes
- Muestreo aleatorio para inspección visual
- Estadísticas descriptivas del dataset

#### **Sección 3: Preprocesamiento y Organización**
```python
# División del dataset
split_ratios = [0.7, 0.15, 0.15]  # train/val/test
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Generadores con augmentación
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
```

#### **Sección 4: Arquitectura y Entrenamiento**
```python
# Transfer Learning con MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False)
base_model.trainable = False

# Cabeza personalizada
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

# Entrenamiento con callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.5),
    ModelCheckpoint("modelo_transfer_parkinson.h5", save_best_only=True)
]
```

#### **Sección 5: Evaluación Exhaustiva**
- Matriz de confusión con visualización
- Métricas médicas (Precision, Recall, F1-Score, AUC-ROC)
- Curva ROC para análisis de discriminación
- Classification report detallado

#### **Sección 6: Pruebas en Tiempo Real**
- Carga de imágenes desde local
- Preprocesamiento automático
- Predicción con interpretación de confianza
- Visualización de resultados

### Configuración del Entorno

```python
# Configuración de GPU
import tensorflow as tf
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))

# Montaje de Google Drive para persistencia
from google.colab import drive
drive.mount('/content/drive')
```

### Dataset Utilizado

- **Fuente**: Kaggle - "Handwritten Parkinson's Disease Augmented Data"  
- **Autor**: banilkumar20phd7071
- **Clases**: Healthy, Parkinson
- **Formato**: Imágenes RGB de trazos de espirales y ondas
- **Preprocesamiento**: Normalización, redimensionamiento, augmentación

## 🐳 Dockerización

### Dockerfile Explicado

```dockerfile
FROM python:3.10-slim          # Imagen base ligera
WORKDIR /app                   # Directorio de trabajo
COPY . .                       # Copiar archivos del proyecto
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requisitos.txt  # Instalar dependencias
EXPOSE 5000                    # Puerto de la aplicación
CMD ["python", "app.py"]  # Comando de inicio
```

### Comandos Docker

```bash
# Construir imagen
docker build -t parkinson-detector .

# Ejecutar contenedor
docker run -p 5000:5000 parkinson-detector

# Ejecutar en segundo plano
docker run -d -p 5000:5000 parkinson-detector

# Ver logs
docker logs <container_id>
```

## 🌐 Despliegue

### Despliegue en AWS EC2

La aplicación está desplegada en **Amazon Web Services EC2**, proporcionando:

- **Alta disponibilidad**: Instancia EC2 con uptime garantizado
- **Escalabilidad**: Capacidad de escalar recursos según demanda  
- **Seguridad**: Grupos de seguridad y configuración SSL
- **Rendimiento**: Optimización para inferencia en tiempo real

### Configuración de EC2

```bash
# Configuración del servidor
Instance Type: t2.micro (Free tier eligible)
Operating System: Ubuntu 20.04 LTS
Security Groups: HTTP (80), HTTPS (443), SSH (22)
Storage: 8GB EBS
```

### Proceso de Despliegue

```bash
# 1. Conectar a la instancia EC2
ssh -i keypair.pem ubuntu@ec2-instance-ip

# 2. Instalar dependencias del sistema
sudo apt update && sudo apt install python3-pip

# 3. Clonar el repositorio
git clone https://github.com/usuario/proyecto-parkinson.git

# 4. Instalar dependencias Python
pip3 install -r requisitos.txt
```

### Plataformas de Despliegue Alternativas

- **Railway**: Despliegue simplificado desde GitHub con CI/CD automático
- **Heroku**: Despliegue gratuito con Git
- **Google Cloud Run**: Serverless containers
- **Azure Container Instances**: Contenedores en la nube
- **DigitalOcean**: VPS económicos

### 🚀 Aplicación Desplegada

La aplicación también está actualmente desplegada y disponible en **Railway**:

**🌐 Demo en vivo**: [https://web-production-1eaa5.up.railway.app/](https://web-production-1eaa5.up.railway.app/)

### 🔧 Despliegue Rápido en Railway

¿Quieres desplegar tu propia instancia? Hazlo desde GitHub:

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/)

**Repositorio alternativo en GitHub**: [devloop307/parkinson-detector](https://github.com/devloop307/parkinson-detector)

**Pasos para desplegar desde GitHub**:

1. **Fork** este repositorio en tu cuenta de GitHub
2. **Conecta** tu repositorio con Railway
3. **Configura** las variables de entorno necesarias
4. **Despliega** automáticamente con cada push

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Para contribuir:

1. **Fork** el repositorio
2. **Crear** una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. **Commit** tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. **Push** a la rama (`git push origin feature/nueva-funcionalidad`)
5. **Crear** un Pull Request

### Áreas de Contribución

- 🔬 Mejoras en el modelo de ML
- 🎨 Diseño de la interfaz
- 📊 Nuevas métricas de evaluación
- 🧪 Tests y validación
- 📖 Documentación
- 🌐 Internacionalización

## ⚕️ Consideraciones Médicas

### ⚠️ Descargo de Responsabilidad

**IMPORTANTE**: Este sistema es una herramienta de apoyo educativa y de investigación. No debe utilizarse como:

- ❌ Único método de diagnóstico
- ❌ Reemplazo de evaluación médica profesional
- ❌ Herramienta de autodiagnóstico definitivo

### 🏥 Recomendaciones

- ✅ Consulte siempre con un neurólogo especializado
- ✅ Use los resultados como información complementaria
- ✅ Considere múltiples evaluaciones y pruebas
- ✅ Mantenga un seguimiento médico regular

### 🔬 Propósito de Investigación

Este proyecto tiene fines:
- Educativos y de investigación
- Desarrollo de herramientas de apoyo
- Avance en neurología computacional
- Contribución a la comunidad científica

## 📄 Licencia

Este proyecto es un trabajo académico desarrollado en la **Universidad Privada Antenor Orrego** para el curso de **Inteligencia Artificial: Principios y Técnicas**.

### 🎓 **Propósito Académico**
- Proyecto educativo sin fines comerciales
- Desarrollo de competencias en Machine Learning
- Aplicación práctica de técnicas de IA en salud
- Contribución al conocimiento en neurología computacional

### 📋 **Términos de Uso**
- El código fuente es de libre consulta para fines educativos
- Se requiere atribución al equipo de desarrollo y universidad
- No se permite uso comercial sin autorización expresa
- El proyecto es de naturaleza investigativa y educativa

---

## 📚 Información Académica

### 🏛️ **Universidad Privada Antenor Orrego**
**Escuela de Ingeniería de Sistemas e Inteligencia Artificial**

### 👨‍🎓 **Equipo de Desarrollo**

| Estudiante | Código |
|------------|--------|
| **ALCÁNTARA RODRÍGUEZ, PIERO ARTURO** | 000268459 |
| **AREVALO ESPINOZA, RAMDHUM** | - |
| **BAUTISTA REYES, LOURDES YOLANDA** | - |
| **DAVALOS ALFARO, MARISELLA LISSET** | - |
| **LEYVA VALQUI, GABRIEL ADOLFO** | 000262610 |
| **RODRIGUEZ GONZALES, ALEJANDRO VALENTINO** | 000267222 |

### 📖 **Curso Académico**
**INTELIGENCIA ARTIFICIAL: PRINCIPIOS Y TÉCNICAS**

### 👨‍🏫 **Docentes**
- **SAGASTEGUI CHIGNE, TEOBALDO HERNAN**
- **MENDOZA CORPUS, CARLOS ALFREDO**

---
