#!/usr/bin/env python
# coding: utf-8

# In[10]:


import json
import numpy as np
import os
import pickle
import joblib
from sklearn.linear_model import LogisticRegression

from azureml.core.model import Model


# In[23]:


def init():
    global model
    # retrieve the path to the pickle file using the model name
    model_path = Model.get_model_path('IRIS')
    model = joblib.load(model_path+"/"+"iris_model.pkl")


# In[21]:


def create_response(predicted_lbl):
    resp_dict = {}
    print("Predicted Species : ", predicted_lbl)
    resp_dict["predicted_species"] = str(predicted_lbl)
    return json.loads(json.dumps({"output": resp_dict}))


# In[22]:


def run(raw_data):
    data = json.loads(raw_data)
    sepal_l_cm = data['SepalLengthCm']
    sepal_w_cm = data['SepalWidthCm']
    petal_l_cm = data['PetalLengthCm']
    petal_w_cm = data['PetalWidthCm']
    predicted_species = model.predict([[sepal_l_cm,sepal_w_cm,petal_l_cm,petal_w_cm]])[0]
    return create_response(predicted_species)