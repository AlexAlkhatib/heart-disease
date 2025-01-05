# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:41:32 2024

@author: Alex Alkhatib
"""

# Import des librairies
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, confusion_matrix, 
    ConfusionMatrixDisplay, 
    classification_report, 
    roc_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import LinearSVC

X_train = pd.read_csv('X_train_clean.csv')
X_test = pd.read_csv('X_test_clean.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# hyperparameters_knn
n_neighbors = 5
weights = 'uniform'
algorithm = 'auto'
leaf_size = 30
p = 2
metric = 'minkowski'

model_knn = KNeighborsClassifier(
    n_neighbors=n_neighbors,
    weights=weights,
    algorithm=algorithm,
    leaf_size=leaf_size,    
    p=p,
    metric=metric
)

model_knn

y_train.values.ravel()


model_knn.fit(X_train, y_train.values.ravel())

y_pred = model_knn.predict(X_test.values)

# comparer les valeurs de prédiction de X_test avec y_test
y_test

# Etudier les metrics 

# Exactitude : Accuracy
acc_score = accuracy_score(y_pred, y_test)

# GridSearchCV (Recherche du meilleur modèle)

# Recherche des meilleurs paramètres
hyperparametres = {
    'n_neighbors' : list(range(3, 20)),
    'weights' : ['uniform', 'distance'],
    'algorithm' : ['ball_tree', 'kd_tree'],
    'p' : list(range(2, 5))    
}

gscv_knn = GridSearchCV(
    estimator=model_knn, 
    param_grid=hyperparametres,
    cv=5,
    scoring= 'f1',
    verbose= 4
)

# lancer le modèle
gscv_knn.fit(X_train, y_train.values.ravel())

# Chercher les meilleurs hyperparamètres correspondants
best_knn = gscv_knn.best_estimator_

best_knn_hyperparameters = gscv_knn.best_params_

best_knn_hyperparameters

# Il y a plus d'exemple dans le train que dans le test
y_pred = best_knn.predict(X_test)

y_pred.shape

y_test.shape

print("Accuracy :", accuracy_score(y_pred, y_test.values))
print("Precision :", precision_score(y_pred, y_test.values))
print("Recall :", recall_score(y_pred, y_test.values))
print("F1 :", f1_score(y_pred, y_test.values))


print(classification_report(y_pred, y_test))

# Instancier un modèle, faire des recherches sur un hyperparamètre

#2. Classification avec LinearSVC
hyperparametres = [
    {
         "penalty" : ["l1", "l2"],
         "loss" : ["squared_hinge"],
         "dual" : [False],
         "C" : [0.1, 1, 2, 3, 10],
         
    }
]

gscv_knn = GridSearchCV(
    estimator= LinearSVC(),
    param_grid= hyperparametres,
    cv = 5,
    scoring= "f1",
    verbose= 4
)

gscv_knn.fit(X_train, y_train.values.ravel())