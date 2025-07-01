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
python aplicacion.py
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
├── 🐍 aplicacion.py             # Aplicación Flask principal
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

#### `aplicacion.py`
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

El modelo `modelov6.h5` implementa una **Red Neuronal Convolucional (CNN)** optimizada para el análisis de patrones gráficos:

```python
# Configuración del modelo
- Capas de entrada: 224x224x3 (RGB)
- Capas convolucionales: Extracción de características
- Capas de pooling: Reducción dimensional
- Capas densas: Clasificación final
- Función de activación: Sigmoid (clasificación binaria)
```

### Preprocesamiento de Imágenes

```python
def predecir_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen)           # Cargar imagen
    img = cv2.resize(img, (224, 224))       # Redimensionar
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB
    img = img / 255.0                       # Normalizar [0,1]
    img = np.expand_dims(img, axis=0)       # Añadir dimensión batch
    return modelo.predict(img)[0][0]        # Predicción
```

### Métricas de Evaluación

- **Precisión**: Porcentaje de predicciones correctas
- **Sensibilidad**: Capacidad de detectar casos positivos
- **Especificidad**: Capacidad de identificar casos negativos
- **F1-Score**: Media armónica entre precisión y recall

## 🔬 Desarrollo en Google Colab

El modelo fue desarrollado y entrenado completamente en **Google Colab**, aprovechando:

- **GPU gratuita**: Aceleración del entrenamiento
- **Entorno preconfigurado**: TensorFlow y dependencias instaladas
- **Almacenamiento en la nube**: Google Drive integration
- **Notebooks interactivos**: Documentación y código unificados

### Proceso de Entrenamiento

1. **Preparación de datos**: Carga y limpieza del dataset
2. **Augmentación**: Técnicas de aumento de datos
3. **División**: Train/Validation/Test splits
4. **Entrenamiento**: Optimización de hiperparámetros
5. **Evaluación**: Métricas de rendimiento
6. **Exportación**: Guardado del modelo final

*Nota: El notebook de Colab con el código completo estará disponible próximamente.*

## 🐳 Dockerización

### Dockerfile Explicado

```dockerfile
FROM python:3.10-slim          # Imagen base ligera
WORKDIR /app                   # Directorio de trabajo
COPY . .                       # Copiar archivos del proyecto
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requisitos.txt  # Instalar dependencias
EXPOSE 5000                    # Puerto de la aplicación
CMD ["python", "aplicacion.py"]  # Comando de inicio
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

### Plataformas de Despliegue

- **Heroku**: Despliegue gratuito con Git
- **AWS EC2**: Instancias escalables
- **Google Cloud Run**: Serverless containers
- **Azure Container Instances**: Contenedores en la nube

### Variables de Entorno

```bash
export FLASK_ENV=production
export FLASK_APP=aplicacion.py
export PORT=5000
```

### URL de la Aplicación

🌍 **Aplicación Desplegada**: [Próximamente]

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

Este proyecto está licenciado bajo la **Licencia MIT** - ver el archivo [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2025 [Tu Nombre]

Se concede permiso, de forma gratuita, a cualquier persona que obtenga una copia
de este software y archivos de documentación asociados...
```

---

## 📞 Contacto y Soporte

- **Desarrollador**: [Tu Nombre]
- **Email**: [tu-email@ejemplo.com]
- **LinkedIn**: [Tu perfil de LinkedIn]
- **Issues**: [GitHub Issues](https://github.com/tu-usuario/proyecto-parkinson/issues)

---

### 🙏 Agradecimientos

- Comunidad de TensorFlow y Keras
- Google Colab por el entorno de desarrollo
- Comunidad médica y de investigación en Parkinson
- Contribuidores de código abierto

---

**¿Te gusta este proyecto? ⭐ ¡Dale una estrella en GitHub!**
