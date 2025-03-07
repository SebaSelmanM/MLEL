# syntax=docker/dockerfile:1.2
#FROM python:latest
# put you docker configuration here
# Usa una imagen base de Python (ej. 3.9)

# Crea un directorio para la app
#WORKDIR /app

# Copia los archivos de requerimientos
#COPY requirements.txt /app

# Instala dependencias (FastAPI, uvicorn, xgboost, etc.)
#RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de tu proyecto
#COPY . /app

# Expone el puerto en que correrá la app dentro del contenedor
#EXPOSE 8080

# Comando por defecto para correr la API con uvicorn
# Ajusta "challenge.api:app" según la ruta donde está tu app

#CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]

# Usa una versión estable de Python (3.9 o 3.10)
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

ENV PYTHONPATH=/usr/local/lib/python3.9/site-packages

# Instalar paquetes del sistema necesarios para construir dependencias
RUN apt-get update && apt-get install -y python3-distutils python3-pip python3-venv build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copiar los archivos de requerimientos
COPY requirements.txt ./

# Instalar las dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir requests

# Copiar el código fuente de la aplicación
COPY . .

# Exponer el puerto 8080
EXPOSE 8080

# Comando para iniciar la API en Cloud Run
#CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
CMD ["python3", "-m", "uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]




