# Aplicación Flask para Titanic

Esta es una aplicación Flask para predecir si un pasajero sobreviviría.

## Paquetes Utilizados:

- Json
- Pandas
- Numpy
- Flask
- WTForms
- Scikit-Learn
- LightGBM

## Pasos:

### Preparación de datos y construcción del modelo

- Lectura de datos
- Limpieza de datos y rellenado de valores faltantes
- División de los datos en datos de entrenamiento y de prueba
- Codificación de predictores categóricos usando LabelEncoder de sklearn
- Selección del mejor modelo a través de la validación cruzada estratificada con accuracy
- Entrenamiento del mejor modelo en el conjunto de entrenamiento completo
- Evaluación del mejor modelo en el conjunto de prueba
- Serialización del modelo en disco

### Predicción en la aplicación Flask

- Navegar a la página de predicción - http://localhost:5000/predict.
- El usuario completa el formulario en el navegador con los valores de los predictores requeridos y luego envía el formulario.
- Los valores proporcionados por los usuarios se introducen en el modelo para obtener predicciones.
- El resultado del modelo se muestra al usuario en el navegador.
- Enviar una solicitud POST a la ruta de API - http://localhost:5000/predict_api.
- Proporcionar los valores de los predictores requeridos en formato JSON.
- Obtener las predicciones y la probabilidad de predicción como respuesta en formato JSON.

## Para ejecutar la aplicación en local

- Abrir terminal en el directorio principal '/Flask_Titanic'.
- Luego ejecutar app.py desde la terminal

```python
python app.py
```

## Para ejecutar la aplicación en Docker

- Abrir terminal en el directorio principal '/Flask_Titanic'.
- Luego ejecutar desde la terminal

```python
docker run -d -p 5000:5000 titanic
```
