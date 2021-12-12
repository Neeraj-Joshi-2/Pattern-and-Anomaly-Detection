# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 09:11:32 2021

@author: NEERAJ
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#%%
X , y = make_classification(n_samples=1000 , n_features=4 , n_classes=2 )

print('Features' , end = '\n\n') 
print(X, end = '\n\n')
print('Labels' , end = '\n\n') 
print(y)

y = np.reshape(y,(1000,1))

#%%
plt.scatter(X[:,0] , y)

#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_x = scaler.fit_transform(X)
print(scaled_x)

#%%
X_train , X_test, y_train, y_test = train_test_split(scaled_x , y , test_size = 0.3)

print('Shape of X_train : ' , X_train.shape)
print('Shape of y_train : ' , y_train.shape)
print('Shape of X_test : ' , X_test.shape)
print('Shape of y_test : ' , y_test.shape)

#%%
model = LogisticRegression()
model.fit(X_train , y_train)

#%%
y_pred = model.predict(X_test)
print('Predicted data' , end = '\n\n')
print(y_pred)


#%%
print('coefficients : ' , model.coef_)
print('intercept : ' , model.intercept_)

#%%
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)
print('Accuracy : ' , score)

#%%
y_pred = np.reshape(y_pred,(300,1))
acc = (1 - np.sum(np.absolute(y_pred - y_test))/y_test.shape[0])*100
print("Accuracy of the model is : ", round(acc, 2), "%")

#%%
from sklearn.metrics import confusion_matrix

print('Confusion Matrix' , end = '\n\n')
print(confusion_matrix(y_test, y_pred))

#%%

y_p = ((X_test[:,0] * model.coef_[: , 0]) + (X_test[:,1] * model.coef_[: , 1]) + (X_test[:,2] * model.coef_[: , 2])
       + (X_test[:,3] * model.coef_[: , 3])) + model.intercept_

y_p = 1 / (1 + np.exp(-y_p))

print(y_p)

#%%
m = y_test.shape[0]

cost = -(1/m) * np.sum((y_test * np.log(y_p)) + ((1 - y_test) * np.log(1 - y_p)))

print("Cost : " , cost)

#%%
y_p = np.reshape(y_p,(300,1))
acc = (1 - np.sum(np.absolute(y_p - y_test))/y_test.shape[0])*100
print("Accuracy of the model is : ", round(acc, 2), "%")














