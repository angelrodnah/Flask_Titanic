# Importando las bibliotecas necesarias
import numpy as np
import pandas as pd
from flask import Flask, render_template, request,redirect, url_for
from wtforms import Form, StringField, validators
import dill

# Preprocesamiento
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# El valor de __name__ debería ser '__main__'
app = Flask(__name__)

# Cargando el modelo y el codificador
with open('./model/model.pkl', 'rb') as model_file:
    model = dill.load(model_file)


# Definiendo una clase de formulario para recuperar la entrada del usuario a través de un formulario
class PredictorsForm(Form):
    p_class = StringField(u'Clase a la que pertenecía el pasajero (Valores válidos: 1, 2, 3)', validators=[validators.input_required()])
    sex = StringField(u'Sexo (0: Femenino y 1: Masculino)', validators=[validators.input_required()])
    age = StringField(u'Edad (Por ejemplo: 24)', validators=[validators.input_required()])
    sibsp = StringField(u'Número de hermanos/cónyuge (Por ejemplo: 3)', validators=[validators.input_required()])
    parch = StringField(u'Número de padres e hijos en el barco.(Valores válidos: 0, 1, 2, 3, 4, 5, 6)', validators=[validators.input_required()])
    fare = StringField(u'Tarifa (Número entero, Por ejemplo: 100)', validators=[validators.input_required()])
    embarked = StringField(u'Puerto en el que embarcó el pasajero (0: Cherbourg; 1: Queenstown; 2:Southampton)', validators=[validators.input_required()])

# Página de índice
@app.route('/')
def index():
    return render_template('index.html')

# Página sobre mí
@app.route('/about')
def about():
    return render_template('about.html')

# Ruta de predicción
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = PredictorsForm(request.form)

    if request.method == 'POST' and form.validate():
        p_class = form.p_class.data
        sex = form.sex.data
        age = form.age.data
        sibsp = form.sibsp.data
        parch = form.parch.data
        fare = form.fare.data
        embarked = form.embarked.data

        # Creando la entrada para las predicciones del modelo
        predict_request = [int(p_class), int(sex), float(age), int(sibsp), int(parch), float(fare), int(embarked)]
        predict_request = np.array(predict_request).reshape(1, -1)

        # Realizando predicciones usando el modelo
        prediction = model.predict(predict_request)[0]
        predict_prob = model.predict_proba(predict_request)[0][1]

        return render_template('predictions.html', prediction=prediction, predict_prob=predict_prob)

    return render_template('predict.html', form=form)

# Ruta de entrenamiento
@app.route('/train', methods=['GET'])
def train():
    # Leyendo los datos
    df = pd.read_csv("./data/titanic.csv")

    # Definiendo las columnas predictoras y de etiqueta
    predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    label = 'Survived'

    # Dividiendo los datos en entrenamiento y prueba
    df_train, df_test, y_train, y_test = train_test_split(df[predictors], df[label], test_size=0.20, random_state=42)

    # Limpieza de datos y rellenado de valores faltantes
    df_train['Age'] = df_train['Age'].fillna(df['Age'].mean())
    df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode()[0])

    # Codificación de variables categóricas
    for column in df_train.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_train[column] = le.fit_transform(df_train[column])

    # Inicializando el modelo
    model = RandomForestClassifier(n_estimators=25, random_state=42)

    # Ajustando el modelo con los datos de entrenamiento
    model.fit(X=df_train, y=y_train)

    # Guardando el modelo entrenado y el codificador en disco
    with open('./model/model.pkl', 'wb') as model_file:
        dill.dump(model, model_file)

    with open('./model/encoder.pkl', 'wb') as encoder_file:
        dill.dump(le, encoder_file)

    return 'Modelo entrenado exitosamente'

if __name__ == '__main__':
    app.run(debug=True)
