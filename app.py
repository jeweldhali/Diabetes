from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('original.html')


@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        Pregnancies = float(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form['BloodPressure'])
        SkinThickness = float(request.form['SkinThickness'])
        Insulin= float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DPF = float(request.form['DiabetesPedigreeFunction'])
        Age = float(request.form['Age'])

        pred_args = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DPF,Age]
        #pred_args_arr = np.array(pred_args)
        #pred_args_arr = pred_args_arr.reshape(1, -1)
        mul_reg = open('model1.pkl','rb')
        ml_model = joblib.load(mul_reg)
        model_predcition = ml_model.predict([pred_args])
        model_predcition1 =1
        if model_predcition == 1:
            res = 'Diabetes'
        else:
            res = 'No Diabetes'
        #return res
    return render_template('predict.html', prediction = res)

if __name__ == '__main__':
    app.run()
