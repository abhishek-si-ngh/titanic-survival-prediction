from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model and encoder
model = joblib.load('titanic_model.pkl')

# Define the expected column names (including the one-hot encoded 'Embarked' columns)
columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    pclass = int(request.form['Pclass'])
    sex = request.form['Sex']
    age = float(request.form['Age'])
    sibsp = int(request.form['SibSp'])
    parch = int(request.form['Parch'])
    fare = float(request.form['Fare'])
    embarked = request.form['Embarked']

    # 🔁 Encode 'Sex': convert 'male' to 0 and 'female' to 1 (or vice versa, depending on your model)
    if sex.lower() == 'male':
        sex_encoded = 0
    else:
        sex_encoded = 1

    # 🔁 Encode 'Embarked' manually into two columns (Q and S)
    if embarked == 'Q':
        embarked_encoded = [1, 0]  # Embarked_Q = 1, Embarked_S = 0
    elif embarked == 'S':
        embarked_encoded = [0, 1]  # Embarked_Q = 0, Embarked_S = 1
    else:
        embarked_encoded = [0, 0]  # Embarked_Q = 0, Embarked_S = 0 (assumed to be C)

    # 📦 Prepare the input as a DataFrame with the correct structure
    input_data = pd.DataFrame([[
        pclass,
        sex_encoded,
        age,
        sibsp,
        parch,
        fare,
        embarked_encoded[0],  # Embarked_Q
        embarked_encoded[1]   # Embarked_S
    ]], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S'])

    # 🎯 Make prediction
    prediction = model.predict(input_data)

    result = 'Survived' if prediction[0] == 1 else 'Did Not Survive'
    return render_template('index.html', prediction_text=f"Prediction: {result}")

    #return render_template('index.html', prediction=prediction[0])


if __name__ == "__main__":
    app.run(debug=True)
