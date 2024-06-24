from flask import Flask,render_template,request
import numpy as np

import pickle
import joblib

app=Flask(__name__)

@app.route("/")
def form():
    return render_template("diabetes.html")

@app.route('/predict',methods=['post'])
def predict():
    value1=int(request.form.get('Pregnancies'))
    value2=int(request.form.get('Glucose'))
    value3=int(request.form.get('BloodPressure'))
    value4=int(request.form.get('SkinThickness'))
    value5=int(request.form.get('Insulin'))
    value6=int(request.form.get('BMI'))
    value7=int(request.form.get('DiabetesPedigreeFunction'))
    value8=int(request.form.get('Age'))

    #print("age=",value8)

    input_data=np.array([[value1,value2,value3,value4,value5,value6,value7,value8]])

    #loading model and prediction

    model=joblib.load("diabetes_model.pkl")
    pred=model.predict(input_data)
    pred_val=int(pred)

    if pred_val==1:
        return "person is diabetic"
    
    return "Person is not diabetic"



 

app.run(debug=True)
