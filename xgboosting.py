#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:15:07 2019
this py is to use xgboosting to select features from data
@author: rain
"""

import numpy as np  
import pandas as pd  
from xgboost import XGBClassifier
import matplotlib.pyplot as plt  

# load data
dataset = pd.read_csv('/Users/rain/Desktop/heart.csv')   # change to real data
# split data into X and y
X = dataset.iloc[:,0:13]  
y = dataset.iloc[:,13]
# fit model no training data  (you can train it into test and train data)
model = XGBClassifier()
x=model.fit(X, y)
from xgboost import plot_importance
plot_importance(model, max_num_features=10) # top 10 most important features
plt.show()




