import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold, train_test_split


# Function to standardize
def standardize(X_train, X_test, X_val):
    mean = X_train.mean()
    std = X_train.std()
    
    X_train_scaled = (X_train - mean)/std
    X_test_scaled = (X_test - mean)/std
    X_val_scaled = (X_val - mean)/std
    
    return X_train_scaled, X_test_scaled, X_val_scaled


#Function to normalize
def min_max_scaler(X_train, X_test, X_val):
    X_max = X_train.max()
    X_min = X_train.min()
    
    X_train_mm = (X_train - X_min) / (X_max - X_min)
    X_test_mm = (X_test - X_min) / (X_max - X_min)
    X_val_mm = (X_val - X_min) / (X_max - X_min)
    
    return X_train_mm, X_test_mm, X_val_mm



    