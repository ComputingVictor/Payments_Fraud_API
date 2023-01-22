# Payments Fraud API

<p align="center">

![image.png](attachment:image.png)


</p>



## Project

This is a project for the subject Machine Learning of CUNEF Master´s in Data Science.

This practice consists of simulating the implementation of the model generated in the fraud practice by means of an API and generating a monitoring dashboard. For this we will have three objectives:

1. Generate a Docker environment to work and to be able to run everything correctly saving the dependencies of operating systems, libraries, environments, etc.
2. To use Flask to be able to invoke the model generated in the previous practice, so that data is passed to it and it returns the prediction.
3. Store in a file of the desired type (csv, json, etc.) all the calls that have been made to the API and the prediction that has been returned, in order to generate a follow-up dashboard.


## What did we use?

- Python 3.9.13
- Visual Studio Code
- Jupyter Notebook
- Flask
- Docker
- Power BI

## How to run the project?

To run the project you could find the full details in `info/Informe_Proyecto`. The summary would be:

- 1. Create the docker image:

`docker build -t docker-api -f Dockerfile .`

- 2. Predict the values:

`docker run docker-api python3 prediction.py`

- 3. Extract the pickle model, replace 'youthful_hermann' with the name of your random container :

`docker cp youthful_hermann:/data/xgb_model_test.pickle .`

- 4. Introduce it in the Flask folder and execute this prompt:

`python app.py`

- 5. You will be able to acces in your local system through the port displayed.


## Content of the repository

- Docker: Utilites to deploy the docker image and obtain the model.

- Flask: Utilites to deploy in local the model, also the BI inform.
    
   - templates: HTML resources. 
   

- info: Step by step guide to follow the repository


## Authors

Victor Viloria Vázquez 
- Email: victor.viloria@cunef.edu
- Linkedin: https://www.linkedin.com/in/vicviloria/


## Project Link:

https://github.com/ComputingVictor/Payments_Fraud_API
