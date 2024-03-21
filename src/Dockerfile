# Usa la imagen base de Python
FROM python:3.11-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el archivo requirements.txt al contenedor
COPY requirements.txt .

# Instala las dependencias del proyecto
RUN pip install -r requirements.txt

# Copia el contenido del directorio actual al contenedor en /app
COPY . .

# Expone el puerto 5000
EXPOSE 5000
RUN apt-get update && apt-get install -y libgomp1
# Comando para ejecutar la aplicación Flask
CMD ["python", "app.py"]
