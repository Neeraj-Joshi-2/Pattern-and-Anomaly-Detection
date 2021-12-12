# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 09:25:05 2021

@author: NEERAJ
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

#%%
X, y = make_regression(1000, 5, noise = 5.0)
NUM_TRIALS = 30

#%%
tuned_parameters = [{'solver' : ['svd', 'lsqr'],'fit_intercept': ['True'],'normalize': ['False']},
                    {'solver' : ['sag', 'cholesky'],'fit_intercept': ['False'],'normalize': ['true']}]
NUM_TRIALS = 30
#%%
score = 'r2'
non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)

#%%

for i in range(NUM_TRIALS):
    x_train = X
    
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    model= GridSearchCV(estimator = linear_model.Ridge(), param_grid = tuned_parameters, scoring = score)
    model.fit(X,y)
    non_nested_scores[i] = model.best_score_
    
    
    # Nested CV with parameter optimization
    model = GridSearchCV(estimator= linear_model.Ridge(), param_grid = tuned_parameters, cv=inner_cv, scoring= score)
    nested_score = cross_val_score(model, X=X, y=y, cv=outer_cv)
    nested_scores[i] = nested_score.mean()
    
#%%  
score_difference = non_nested_scores - nested_scores

print("Average difference of {:6f} with std. dev. of {:6f}.".format(score_difference.mean(), score_difference.std()))

#%%

#for Logistic Regression

#%%
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=1000 , n_features=4 , n_classes=2)
NUM_TRIALS = 30

#%%
tuned_parameters = [{'fit_intercept':[True],'solver':['lbfgs']},
                    {'fit_intercept':[False],'solver':['saga']}]

#%%

non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)

#%%
for i in range(NUM_TRIALS):
    x_train = X
    
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    model=GridSearchCV(LogisticRegression(),tuned_parameters,scoring = 'accuracy')
    model.fit(X,y)
    non_nested_scores[i] = model.best_score_
    
    
    # Nested CV with parameter optimization
    model = GridSearchCV(estimator= LogisticRegression(), param_grid = tuned_parameters, cv=inner_cv, scoring = 'accuracy')
    nested_score = cross_val_score(model, X=X, y=y, cv=outer_cv)
    nested_scores[i] = nested_score.mean()
    
#%%  

score_difference = non_nested_scores - nested_scores

print("Average difference of {:6f} with std. dev. of {:6f}.".format(score_difference.mean(), score_difference.std()))

#%%

#for Breast Cancer Detection

#%%

from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
print('Features\n')
print(X)
print()
print('Labels\n')
print(y)

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print('Shape of X_train : ' , X_train.shape)
print('Shape of y_train : ' , y_train.shape)
print('Shape of X_test : ' , X_test.shape)
print('Shape of y_test : ' , y_test.shape)
#%%
NUM_TRIALS = 30

tuned_parameters = [{'fit_intercept':[True],'solver':['lbfgs']},
                    {'fit_intercept':[False],'solver':['saga']}]

#%%

non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)

#%%

for i in range(NUM_TRIALS):
    x_train = X_train
    
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    model=GridSearchCV(LogisticRegression(),tuned_parameters,scoring = 'accuracy')
    model.fit(X_train,y_train)
    non_nested_scores[i] = model.best_score_
    
    
    # Nested CV with parameter optimization
    model = GridSearchCV(estimator= LogisticRegression(), param_grid = tuned_parameters, cv=inner_cv, scoring = 'accuracy')
    model.fit(X_train,y_train)
    nested_score = cross_val_score(model, X=X_train, y=y_train, cv=outer_cv)
    nested_scores[i] = nested_score.mean()

#%%

score_difference = non_nested_scores - nested_scores

print("Average difference of {:6f} with std. dev. of {:6f}.".format(score_difference.mean(), score_difference.std()))

#%%
y_pred = model.predict(X_test)
print(y_pred)

#%%
from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_pred))

#%%
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)
print('Accuracy : ' , score*100)

#%%
Average difference of -0.003602 with std. dev. of 0.005601.
Accuracy :  96.49122807017544