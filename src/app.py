# Importing necessary libraries
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from wtforms import Form, StringField, validators
import dill

# Preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Value of __name__ should be '__main__'
app = Flask(__name__)

# Loading model and encoder
with open('./model/model.pkl', 'rb') as model_file:
    model = dill.load(model_file)


# Defining a form class to retrieve input from the user through a form
class PredictorsForm(Form):
    p_class = StringField(u'P Class (Valid Values: 1, 2, 3)', validators=[validators.input_required()])
    sex = StringField(u'Sex (0: Female and 1: Male)', validators=[validators.input_required()])
    age = StringField(u'Age (For example: 24)', validators=[validators.input_required()])
    sibsp = StringField(u'Siblings and Spouse Count (For example: 3)', validators=[validators.input_required()])
    parch = StringField(u'Parch (Valid Values: 0, 1, 2, 3, 4, 5, 6)', validators=[validators.input_required()])
    fare = StringField(u'Fare (For example: 100)', validators=[validators.input_required()])
    embarked = StringField(u'Embarked (Valid Values: 0, 1, 2)', validators=[validators.input_required()])

# Index page
@app.route('/')
def index():
    return render_template('index.html')

# About me page
@app.route('/about')
def about():
    return render_template('about.html')

# Prediction route
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

        # Creating input for model predictions
        predict_request = [int(p_class), int(sex), float(age), int(sibsp), int(parch), float(fare), int(embarked)]
        predict_request = np.array(predict_request).reshape(1, -1)

        # Making predictions using the model
        prediction = model.predict(predict_request)[0]
        predict_prob = model.predict_proba(predict_request)[0][1]

        return render_template('predictions.html', prediction=prediction, predict_prob=predict_prob)

    return render_template('predict.html', form=form)

# Training route
@app.route('/train', methods=['GET'])
def train():
    # Reading data
    df = pd.read_csv("./data/titanic.csv")

    # Defining predictors and label columns
    predictors = ['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    label = 'Survived'

    # Splitting data into training and testing
    df_train, df_test, y_train, y_test = train_test_split(df[predictors], df[label], test_size=0.20, random_state=42)

    # Data cleaning and filling missing values
    df_train.Age = df_train.Age.fillna(df.Age.mean())
    df_train.Embarked = df_train.Embarked.fillna(df_train.Embarked.mode()[0])

    # Encoding categorical variables
    for column in df_train.columns:
        if df_train[column].dtype == np.object:
            le = LabelEncoder()
            df_train[column] = le.fit_transform(df_train[column])

    # Initializing the model
    model = RandomForestClassifier(n_estimators=25, random_state=42)

    # Fitting the model with training data
    model.fit(X=df_train, y=y_train)

    # Saving the trained model and encoder on disk
    with open('./model/model.pkl', 'wb') as model_file:
        dill.dump(model, model_file)

    with open('./model/encoder.pkl', 'wb') as encoder_file:
        dill.dump(le, encoder_file)

    return 'Model trained successfully'

if __name__ == '__main__':
    app.run(debug=True)

