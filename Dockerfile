# Usa la imagen base de Python
FROM python:3.11-slim

# Copia el archivo requirements.txt al contenedor
COPY requirements.txt .

# Instala las dependencias del proyecto
RUN pip install -r requirements.txt

# Copia el contenido de la carpeta actual al contenedor en /src
COPY . /src

# Establece el directorio de trabajo en /src
WORKDIR /src

# Expone el puerto 5000
EXPOSE 5000

# Comando para ejecutar la aplicación Flask
CMD ["python", "app.py"]
