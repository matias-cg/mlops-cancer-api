# Usa una imagen base oficial de Python 3.11 ligera (slim)
FROM python:3.11-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 1. Copiar dependencias e instalarlas PRIMERO para optimizar el caché de capas de Docker.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copiar y ejecutar el script de entrenamiento para garantizar la reproducibilidad del modelo.
# Esto asegura que el modelo se genera con las bibliotecas instaladas en el contenedor.
COPY train_model.py .
RUN python train_model.py

# 3. Copiar la aplicación de la API
COPY app.py .

# 4. Exponer el puerto
EXPOSE 5000

# 5. Comando para ejecutar la aplicación con Gunicorn (servidor WSGI robusto y de alto rendimiento)
# Esto es esencial para el despliegue profesional.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "60", "app:app"]
