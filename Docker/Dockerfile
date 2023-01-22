FROM jupyter/scipy-notebook

WORKDIR /data

RUN pip install xgboost
RUN pip install category_encoders
RUN pip install plotly

COPY data.csv ./data.csv
COPY aux_functions.py ./aux_functions.py

COPY train.py ./train.py
COPY prediction.py ./prediction.py

RUN python3 train.py
