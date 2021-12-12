# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 09:47:49 2021

@author: NEERAJ
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
#%%
X , y = make_classification(n_samples=100 , n_features=10,n_informative=3 , n_classes=2)
print('shape of X : ' , X.shape)
print('shape of y : ' , y.shape)
#%%
pca = PCA(n_components=3)
pca.fit(X)
x_new = pca.transform(X)
print('shape of new_x : ' , x_new.shape)
#%%
plt.scatter(x_new[:,0] , y)
#%%
X_train , X_test, y_train, y_test = train_test_split(x_new , y , test_size = 0.1)

print('Shape of X_train(with PCA) : ' , X_train.shape)
print('Shape of y_train : ' , y_train.shape)
print('Shape of X_test(with PCA) : ' , X_test.shape)
print('Shape of y_test : ' , y_test.shape)
#%%
model = LogisticRegression()
model.fit(X_train , y_train)
y_pred = model.predict(X_test)
print('Predicted data' , end = '\n\n')
print(y_pred)
#%%
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)
print('Accuracy with PCA : ' , score)
#%%
X_train_2 , X_test_2, y_train_2, y_test_2 = train_test_split(X , y , test_size = 0.2)

print('Shape of X_train(without PCA) : ' , X_train_2.shape)
print('Shape of y_train : ' , y_train_2.shape)
print('Shape of X_test(without PCA) : ' , X_test_2.shape)
print('Shape of y_test : ' , y_test_2.shape)

#%%
model.fit(X_train_2 , y_train_2)
y_pred_2 = model.predict(X_test_2)
print('Predicted data' , end = '\n\n')
print(y_pred_2)
#%%
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test_2 , y_pred_2)
print('Accuracy without PCA: ' , score)

#%%

X1 , y1 = make_classification(n_samples=100 , n_features=10,n_informative=7 , n_classes=2)

#%%
pca1 = PCA(n_components=9)
pca1.fit(X1)
x_new_1 = pca.transform(X1)
X_train1 , X_test1, y_train1 , y_test1 = train_test_split(x_new_1 , y1 , test_size = 0.1)
model.fit(X_train1 , y_train1)
y_pred1 = model.predict(X_test1)

pca2 = PCA(n_components=5)
pca2.fit(X1)
x_new_2 = pca.transform(X1)
X_train2 , X_test2, y_train2 , y_test2 = train_test_split(x_new_2 , y1 , test_size = 0.1)
model.fit(X_train2 , y_train2)
y_pred2 = model.predict(X_test2)

pca3 = PCA(n_components=3)
pca3.fit(X1)
x_new_3 = pca.transform(X1)
X_train3 , X_test3, y_train3 , y_test3 = train_test_split(x_new_3 , y1 , test_size = 0.1)
model.fit(X_train3 , y_train3)
y_pred3 = model.predict(X_test3)

#%%

score1 = accuracy_score(y_test1 , y_pred1)
print('Accuracy (9) : ' , score1)
score2 = accuracy_score(y_test2 , y_pred2)
print('Accuracy (5) : ' , score2)
score3 = accuracy_score(y_test3 , y_pred3)
print('Accuracy (3): ' , score3)

#%%

























