#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import joblib


# In[4]:


def load_preprocessed_data():
    X_train, y_train, X_test, y_test = (pd.read_csv(f"../Data/Processed_Data/{file}.csv") for file in ["X_train", "y_train", "X_test", "y_test"])
    return X_train, y_train, X_test, y_test


# In[5]:


def fit_and_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    
    model_scores={}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name]=model.score(X_test, y_test)
        
    return model_scores


# In[ ]:


def save_model(model, filename):
    joblib.dump(model, filename)

