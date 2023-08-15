#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt
import joblib


# In[13]:


def load_test_data():
    X_test, y_test = (pd.read_csv(f"../Data/Processed_Data/{file}.csv") for file in ["X_test", "y_test"])
    return X_test, y_test


# In[14]:


def load_trained_model(model_filename):
    model = joblib.load(model_filename)
    return model


# In[15]:


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    return accuracy, classification_rep, conf_matrix, precision, recall, f1_score, fpr, tpr, roc_auc


# In[44]:


def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.show()

