# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:55:17 2024

@author: Alex Alkhatib
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


df = pd.read_csv('Heart Disease (3).csv', index_col=0)

num_variables = ["age", "trestbps", "chol", "thalach", "ca", "oldpeak"]

def set_num(x):
    if x == 0:
        return 0
    else:
        return 1
    
df["target"] = df["num"].apply(set_num)

# 5. Séparer le dataset en un jeu d'entraînement/validation et un jeu de test. 
# On utilisera la fonction train_test_split.

X = df.drop(["target", "num"], axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = df["target"])

num_pipeline = Pipeline(steps = [
    ("imputer", SimpleImputer()),
    ("normalization", MinMaxScaler())
])

preprocessor = ColumnTransformer(transformers=
     [
        ("numeric", num_pipeline, num_variables)
     ]
)

preprocessor

X_train_clean = preprocessor.fit_transform(X_train)

X_test_clean = preprocessor.transform(X_test)

pd.DataFrame(X_train_clean).to_csv("./X_train_clean.csv", index=False)
pd.DataFrame(X_test_clean).to_csv("./X_test_clean.csv", index=False)
pd.DataFrame(y_train).to_csv("./y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("./y_test.csv", index=False)

X_train = pd.read_csv('X_train_clean.csv')
X_test = pd.read_csv('X_test_clean.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')