# Aplicación Flask para Titanic

Esta es una aplicación Flask para predecir si un pasajero sobreviviría.

## Paquetes Utilizados:

```
+ Json
+ Pandas
+ Numpy
+ Flask
+ WTForms
+ Scikit-Learn
```

## Pasos:

### Preparación de datos y construcción del modelo

- Lectura de datos
- División de los datos en datos de entrenamiento y de prueba
- Imputación de valores faltantes del conjunto de datos de entrenamiento.
- Uso de valores imputados del conjunto de datos de entrenamiento para llenar valores faltantes en el conjunto de datos de prueba para evitar fugas de datos.
- Codificación de predictores categóricos usando LabelEncoder de sklearn
- Seleccion del modelo a traves de baseline
- Serialización del modelo en disco

### Predicción en la aplicación Flask

```
+ Navegar a la página de predicción - http://localhost:5000/predict.
+ El usuario completa el formulario en el navegador con los valores de los predictores requeridos y luego envía el formulario.
+ Los valores proporcionados por los usuarios se introducen en el modelo para obtener predicciones.
+ El resultado del modelo se muestra al usuario en el navegador.
```

## Para ejecutar la aplicación

- Abrir terminal en el directorio principal '/Flask_Titanic'.
- Luego ejecutar app.py desde la terminal

```
python app.py
```

Esto iniciará el servidor de la aplicación Flask. Ahora puede abrir el sitio abriendo http://localhost:5000 `<br>`

Nota: la aplicación se ejecuta en modo de depuración
