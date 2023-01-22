import os
import csv
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import xgboost as xgb

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

app = Flask(__name__)
model = pickle.load(open('xgb_model_test.pickle', 'rb'))


@app.route('/')
def index():
    return render_template('form.html') 

@app.route('/predict',methods=['POST'])
def predict():

    data = pd.DataFrame(np.array(list(request.form.values())).reshape(1, -1), columns=['step', 'type', 'amount', 'device', 'connection_time','nameOrig', 'gender',
                                                                    'oldbalanceOrg', 'age','newbalanceOrig','zone', 'user_number',
                                                                    'nameDest','user_connections','security_alert','oldbalanceDest', 'newbalanceDest'])


    pred = model.predict_proba(data)[:,1]

   
    
    output = (pred[0]) * 100

    data['prediction'] = output

    if output > 50:
        data['Fraud'] = 1
    else:
        data['Fraud'] = 0

    if os.path.isfile('log.csv'):

        data.to_csv('log.csv', mode='a', header=False, index=False,sep=';',decimal='.')
    else:

        data.to_csv('log.csv', mode='w', index=False,sep=';',decimal='.')

 


    return render_template('form.html', 
                             prediction_text=' La probabilidad de que la operación sea fraude es del',
                             prediction_prob= str(round(output,2)) + ' %')



if __name__ == "__main__":
    app.run(debug=True)