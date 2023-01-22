#!/usr/bin/python3
# inference.py
# Xavier Vasques 13/04/2021


import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import numpy as np
from aux_functions import *
import pandas as pd
import math
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
import category_encoders as ce



def prediction():
    dirpath = os.getcwd()
    print("dirpath = ", dirpath, "\n")
    output_path = os.path.join(dirpath,'output.xlsx')
    print(output_path,"\n")

    x_test = pd.read_csv('x_test.csv')
    y_test = pd.read_csv('y_test.csv')

    # Load the modelo

    xgb_model_test = load_model('xgb_model_test.pickle')


    print("Model score and classification:")

   

    # Predict the probabilities of the target value for the validation set.

    predict_probabilities = xgb_model_test.predict_proba(x_test)
    print(pd.DataFrame(xgb_model_test.predict_proba(x_test)[:,1]))




    pd.DataFrame(xgb_model_test.predict_proba(x_test)[:,1]).to_excel(output_path)
    
    
if __name__ == '__main__':
    prediction()
