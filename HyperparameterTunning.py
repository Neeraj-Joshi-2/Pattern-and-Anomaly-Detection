# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 09:21:25 2021

@author: NEERAJ
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
#%%
x,y = make_regression(n_samples = 1000, n_features=4)

print('Shape of x : ' , x.shape)
print('Shape of y : ' , y.shape)
y = np.reshape(y,(1000,1))
print('New shape of y : ' , y.shape)
#%%
plt.scatter(x[:,1], y)

#%%

tuned_parameters = [{'fit_intercept': ['True'], 'normalize': ['True']},{'fit_intercept': ['False'], 'normalize': ['True']}]

#%%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print('Shape of X_train : ' , X_train.shape)
print('Shape of y_train : ' , y_train.shape)
print('Shape of X_test : ' , X_test.shape)
print('Shape of y_test : ' , y_test.shape)

#%%
model = GridSearchCV(LinearRegression(),tuned_parameters,scoring=('r2'))
model.fit(X_train,y_train)

#%%
print("Best parameters : ", model.best_params_)
print("Best Score : ",model.best_score_)

#%%
print(model.cv_results_)

#%%
from sklearn import metrics

x , y = make_classification(n_samples=1000 , n_features=4 , n_classes=2)

y = y.reshape([1000 , 1])
#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler .fit_transform(x)

X_train , X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print('Shape of X_train : ' , X_train.shape)
print('Shape of y_train : ' , y_train.shape)
print('Shape of X_test : ' , X_test.shape)
print('Shape of y_test : ' , y_test.shape)

y = y.reshape([1000 , 1])
#%%
plt.scatter(X_train[:,1],y_train)
plt.show()

#%%
model = LogisticRegression()
model.fit(X_train,y_train)
pred_y = model.predict(X_test)
print(pred_y)

#%%

print(metrics.confusion_matrix(y_test, pred_y))

#%%

tuned_parameters = [{'fit_intercept':[True],'solver':['lbfgs']},
                    {'fit_intercept':[False],'solver':['saga']}]

#%%
model=GridSearchCV(LogisticRegression(),tuned_parameters,scoring = 'accuracy')
model.fit(X_train , y_train)
#%%

print(model.best_params_ , "\n")
print(model.cv_results_)