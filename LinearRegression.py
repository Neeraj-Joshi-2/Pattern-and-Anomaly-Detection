# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 09:21:59 2021

@author: NEERAJ
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#%%
x,y = make_regression(n_samples = 1000, n_features=1, shuffle = True)

print('Shape of x : ' , x.shape)
print('Shape of y : ' , y.shape)
y = np.reshape(y,(1000,1))
print('New shape of y : ' , y.shape)

plt.scatter(x, y)

#%%
noise_x = np.random.normal(0, 0.5, x.shape)
print('Shape of noise_x : ' , noise_x.shape)

noise_y = np.random.normal(0, 0.5, y.shape)
print('Shape of noise_y : ' , noise_y.shape)

#%%
x = x + noise_x
#y = y + noise_y
plt.scatter(x, y)

#%%

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)
plt.scatter(scaled_x, y)

#%%
X_train, X_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.2)

print('Shape of X_train : ' , X_train.shape)
print('Shape of y_train : ' , y_train.shape)
print('Shape of X_test : ' , X_test.shape)
print('Shape of y_test : ' , y_test.shape)


#%%

reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
#%%
print(y_pred)
#%%
print('Slope using sklearn: ' , reg.coef_)
print('Intercept using sklearn: ' , reg.intercept_)
print('Score : ' , reg.score(X_test , y_test))

#%%

plt.scatter(X_train , y_train)

plt.plot(X_train , reg.coef_* X_train + reg.intercept_ , '-r')


#%%

mean_x = X_train.mean()
mean_y = y_train.mean()

temp_1 = ((X_train - mean_x) * (y_train - mean_y)).sum()
temp_2 = ((X_train - mean_x)**2).sum()

slope = temp_1 / temp_2

print("Slope from manual calculation : " , slope)


intercept = mean_y - (slope * mean_x)
print("Intercept from manual calculation : " , intercept)

#%%

y_pred_2 = (slope * X_test) + intercept
print(y_pred_2)

#%%

df = pd.DataFrame(y_pred)

df[''] = y_pred_2

df.columns = ['SkLearn' , 'Manual']



print(df)





























