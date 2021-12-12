import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV 

#%%

# we create 200 separable points
X, y = make_blobs(n_samples=200, centers=2, random_state=6)
print(X)
print(y)

# fit the model, don't regularize for illustration purposes

#%%
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=0)
print('Shape of X_train : ' , X_train.shape)
print('Shape of y_train : ' , y_train.shape)
print('Shape of X_test : ' , X_test.shape)
print('Shape of y_test : ' , y_test.shape)

#%%

#SVM classifier without cross validation and hyperparameter tuning
clf = svm.SVC(kernel="linear", C=1000)

#SVM classifier with cross validation
clf_cross_val = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf_cross_val, X, y, cv=5)

#model with hyperparameter tuning

from sklearn.svm import SVC
model = SVC()

param_grid = {'C' : [0.1, 1, 10, 100, 1000],
           'gamma' : [1, 0.1, 0.01, 0.001, 0.0001],
           'kernel' : ['rbf']}

grid = GridSearchCV(SVC() , param_grid , refit = 'True', verbose = 3)


#%%

clf.fit(X_train, y_train)
print('Score with simple model : ' , clf.score(X_test, y_test) , end='\n\n')

print('score with cross validation' , scores , end='\n\n')
print("%0.2f accuracy " % (scores.mean()) , end='\n\n')

grid.fit(X,y)
print('Best parameters : ' , grid.best_params_ , end='\n\n')
print('Best estimator : ' , grid.best_estimator_)



#%%

from sklearn.metrics import confusion_matrix

y_pred = grid.predict(X_test)

print('Confusion Matrix' , end = '\n\n')
print(confusion_matrix(y_test, y_pred))