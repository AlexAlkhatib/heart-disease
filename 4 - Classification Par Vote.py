# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:47:22 2024

@author: Alex Alkhatib
"""
'''

Pour mettre en place une classification par vote, appelée également "voting classifier", vous allez combiner plusieurs modèles de machine learning de natures différentes pour améliorer la robustesse et la précision de votre prédiction. En utilisant la librairie scikit-learn, cela peut être réalisé facilement avec la classe VotingClassifier. Voici comment vous pourriez procéder étape par étape :

1. Importer les librairies nécessaires

'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

'''
2. Préparer les données
Chargez vos données, réalisez le prétraitement nécessaire, et divisez-les en jeux d'entraînement et de test. Assurez-vous que toutes les transformations (comme l'imputation des valeurs manquantes et la normalisation) sont bien intégrées dans un pipeline pour éviter les fuites de données.

'''

# Charger les données
df = pd.read_csv('HeartDiseaseUCI.csv')

# Prétraitement des données
X = df.drop(['target'], axis=1)
y = df['target']

# Diviser les données en jeux d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline de prétraitement pour les données numériques
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Appliquer le prétraitement
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

'''
3. Sélectionner et configurer les modèles
Choisissez des modèles variés pour la diversité. Par exemple, vous pouvez combiner un modèle linéaire, un modèle basé sur des arbres et un modèle à vecteurs de support.

'''

# Configurer les modèles individuels
model1 = LogisticRegression(random_state=42)
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model3 = SVC(probability=True, random_state=42)

# Créer le classificateur par vote
voting_clf = VotingClassifier(
    estimators=[('lr', model1), ('rf', model2), ('svc', model3)],
    voting='soft'  # 'hard' pour le vote majoritaire, 'soft' pour le vote basé sur les probabilités
)

'''
4. Entraîner le classificateur par vote
Entraînez le classificateur par vote sur le jeu d'entraînement et évaluez-le sur le jeu de test.

'''

# Entraîner le modèle
voting_clf.fit(X_train, y_train)

# Évaluer le modèle
y_pred = voting_clf.predict(X_test)
print(classification_report(y_test, y_pred))

'''
5. Analyser les résultats
Après avoir entraîné et évalué votre modèle, analysez les résultats pour comprendre les performances du modèle combiné et comment il compare aux modèles individuels.

Cette approche peut vous aider à tirer le meilleur parti des forces de différents modèles tout en atténuant leurs faiblesses, améliorant ainsi la stabilité et la précision de vos prédictions.
'''