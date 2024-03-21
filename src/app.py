# Importando las bibliotecas necesarias
import numpy as np
import pandas as pd
from flask import Flask, render_template, request,redirect, url_for
from wtforms import Form, StringField, validators
import dill
from flask import jsonify

# Preprocesamiento
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
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

#Ruta de api
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()

    p_class = data.get('p_class')
    sex = data.get('sex')
    age = data.get('age')
    sibsp = data.get('sibsp')
    parch = data.get('parch')
    fare = data.get('fare')
    embarked = data.get('embarked')

    # Verificar si se proporcionaron todos los campos necesarios
    if None in [p_class, sex, age, sibsp, parch, fare, embarked]:
        return jsonify({'error': 'Se deben proporcionar todos los campos necesarios.'}), 400

    # Crear la entrada para las predicciones del modelo
    predict_request = [int(p_class), int(sex), float(age), int(sibsp), int(parch), float(fare), int(embarked)]
    predict_request = np.array(predict_request).reshape(1, -1)

    # Realizar predicciones usando el modelo
    prediction = model.predict(predict_request)[0]
    predict_prob = model.predict_proba(predict_request)[0][1]

    # Devolver los resultados como JSON
    return jsonify({'prediction': prediction, 'predict_prob': predict_prob})


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
    
    # Limpieza de datos y rellenado de valores faltantes
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df['Pclass'] = df['Pclass'].fillna(df['Pclass'].mode()[0])
    df['Sex'] = df['Sex'].fillna(df['Sex'].mode()[0])
    
    # Definiendo las columnas predictoras y de etiqueta
    predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    label = 'Survived'
    X = df[predictors].copy()
    y= df[label].copy()
    
    # Dividiendo los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42,stratify=y,shuffle=True)

    # Definir los modelos
    models = {
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(probability=True),
        'LightGBM': LGBMClassifier()}

    # Inicializar variables para almacenar el mejor modelo y su puntuación de precisión
    best_model = None
    best_accuracy = 0.0

    # Realizar validación cruzada estratificada en cada modelo y encontrar el mejor
    for name, model in models.items():
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
        avg_accuracy = cv_scores.mean()


        # Actualizar el mejor modelo si se encuentra uno con mejor precisión
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_model = model

    # Entrenar el mejor modelo en el conjunto de entrenamiento completo
    best_model.fit(X_train, y_train)
    # Evaluación del mejor modelo en el conjunto de prueba
    test_accuracy = best_model.score(X_test, y_test)
    
    # Guardando el modelo entrenado y el codificador en disco
    with open('./model/model.pkl', 'wb') as model_file:
        dill.dump(model, model_file)


    return 'Modelo entrenado exitosamente con precisión de {}%'.format(round(test_accuracy*100, 2))

if __name__ == '__main__':
    app.run(debug=True)
