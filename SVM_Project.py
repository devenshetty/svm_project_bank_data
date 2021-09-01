# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 14:56:43 2021

@author: Deven Shetty
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading data from the csv file
bank_data = pd.read_csv("bill_authentication.csv")

#viewing the data
print(bank_data.shape)
print(bank_data)

#data preprocessing

#(i) separating the attributes and labels
X = bank_data.drop("Class", axis=1)
y = bank_data["Class"]


#(ii) splitting dataset into training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20 )

#training the model
#using simple linear svm
from sklearn.svm import SVC
classifier = SVC(kernel="linear")
classifier.fit(X_train, y_train)

#predictions
y_pred = classifier.predict(X_test)

#evaluating the performance of the model
from sklearn.metrics import  confusion_matrix
print(confusion_matrix(y_test,y_pred))









