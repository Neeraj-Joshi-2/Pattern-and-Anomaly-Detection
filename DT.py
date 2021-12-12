import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#%%
X , y = make_classification(n_samples=1000 , n_features=10 , n_informative=7 , n_classes=3)

print('Features' , end = '\n\n') 
print(X, end = '\n\n')
print('Labels' , end = '\n\n') 
print(y)

#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
plt.scatter(X[:, 0], y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('Shape of X_train : ' , X_train.shape)
print('Shape of y_train : ' , y_train.shape)
print('Shape of X_test : ' , X_test.shape)
print('Shape of y_test : ' , y_test.shape)

#%%
clf = RandomForestClassifier()

param_grid = {  
    'bootstrap': [True], 
    'max_depth': [5, 10, None], 
    'max_features': ['auto', 'log2'], 
    'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]}


grid = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 3, verbose = 2)




#%%

clf.fit(X_train , y_train)
pred = clf.predict(X_test)

grid.fit(X_train , y_train)
y_pred = grid.predict(X_test)



#%%
score = accuracy_score(y_test,pred)
print('Accuracy with normal random forest classifier : ' , score , end = '\n\n')

score = accuracy_score(y_test,y_pred)
print('Accuracy with tuned random forest classifier : ' , score , end = '\n\n')

print('Best parameters : ' , end = '\n\n')
print(grid.best_params_)

from sklearn.metrics import confusion_matrix

print('Confusion Matrix' , end = '\n\n')
print(confusion_matrix(y_test, y_pred))


#%%





