

import platform; print(platform.platform()) 
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
import category_encoders as ce
import math
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
from aux_functions import *

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def f2_func(y_true, y_pred):
    f2_score = fbeta_score(y_true, y_pred, beta=2)
    return f2_score


def train():

    ruta = "data.csv"
    payments_df = pd.read_csv(ruta, sep=';')

    # Transform the decimal format of the variable 'connection_time' by replacing to dots.
    # Transform the data type converting it as float.

    payments_df = payments_df.assign(**{'connection_time': lambda df: df['connection_time'].str.replace(',', '.').astype(float)})

    # Transform the categorical variables 'security_alert', 'isFraud' as object.    

    payments_df = (payments_df
        .assign(**{'security_alert': lambda df: df['security_alert'].astype(object)})
        .assign(**{'isFraud': lambda df: df['isFraud'].astype(object)})
                )

    payments_df = payments_df.drop(['race'],axis = 'columns')

    # Loop to input the first letter of each row from the 'nameOrig' variable.
    letter_orig = []
    for row in payments_df.nameOrig:
        letter_orig.append(row[0])
        
    # Loop to input the first letter of each row from the 'nameDest' variable.   
    letter_dest = []
    for row in payments_df.nameDest:
        letter_dest.append(row[0])
        
    # Substitution of the ID's from  'nameOrig', 'nameDest' for their first letter. 
    payments_df['nameOrig'] = letter_orig
    payments_df['nameDest'] = letter_dest

    # Creation of Pipelines.

    # Creation a list with the name of the categorical variables.

    categ_var = ['type', 'gender', 'device', 'zone','security_alert','nameDest', 'nameOrig', 'nameDest']

    # Creation a list with the name of the numerical variables.

    num_var = ['step', 'amount', 'connection_time', 'oldbalanceOrg','age', 'newbalanceOrig', 'user_number','oldbalanceDest','newbalanceDest']

    # Creation of a pipeline for categorical variables that fill NaN values to 'unknow' first,
    # then apply the OneHotEncoder preprocessing.
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknow')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Creation of a pipeline for numerical variables that standardize each variable

    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

        
    # Creation of a preprocessor for the categorical and numerical variables

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, categ_var),
            ('num', num_transformer, num_var)
        ]
    )

    # Separation of variables X  from the target variable Y 

    X = payments_df.drop('isFraud', axis=1)
    Y = payments_df['isFraud']

    # Division into train and test sets.

    x_train_all, x_test, y_train_all, y_test = train_test_split(X, Y, test_size=0.25, random_state=12345, stratify=Y)

    # Creation of a validation set shuffled from the training set. Selecting a 25% of the total for the 'val_df'.

    x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.25, random_state=12345, stratify=y_train_all)

  
    x_test.to_csv("x_test.csv")
    y_test.to_csv("y_test.csv")

    # Design the parameters.

    n_jobs = -1
    random_state = 12345
    max_depth = 5
    verbosity = 0
    # Create a pipeline that applies a preprocess of the data and then the LogisticRegression model.

    xgb_model_test = Pipeline([
        ('preprocessor', preprocessor),
        ('clasificador', XGBClassifier(n_jobs=n_jobs, random_state=random_state, max_depth=max_depth, verbosity=verbosity)
        )
    ])

    # Optimization of the xgboost_model with the train set: input and target data. 

    xgb_model_test.fit(x_train_all, y_train_all)

    # Save the backup of the XGB optimziation as a pickle file.

    save_model(xgb_model_test,'xgb_model_test.pickle')
    
        
if __name__ == '__main__':
    train()
