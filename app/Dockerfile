# syntax=docker/dockerfile:1

# Usa la imagen base de Python
FROM python:3.11-slim

# Establece el directorio de trabajo en /app
WORKDIR /Flask_Titanic

# Copia el archivo requirements.txt al contenedor
COPY requirements.txt /app

# Instala las dependencias del proyecto
RUN pip install -r requirements.txt

# Copia el contenido del directorio actual al contenedor en /app
COPY . /app

# Expone el puerto 5000
EXPOSE 5000
RUN apt-get update && apt-get install -y libgomp1
# Comando para ejecutar la aplicación Flask
ENV FLASK_APP=app.py
CMD ["flask", "run", "--host", "0.0.0.0"]
