FROM tensorflow/tensorflow:2.19.0

WORKDIR /app

COPY . .

# Instala dependencias ignorando paquetes conflictivos y evitando errores de espacio
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --ignore-installed \
    numpy==1.26.4 \
    flask \
    opencv-python-headless

EXPOSE 5000

CMD ["python", "app.py"]
