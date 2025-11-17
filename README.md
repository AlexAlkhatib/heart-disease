# ❤️ **Cardiovascular Risk Prediction — Classification Models & Ensemble Learning**

Ce projet a pour objectif de **prédire les risques cardiovasculaires** à partir de données médicales en utilisant plusieurs modèles de classification.
Il s’agit d’un projet **personnel**, conçu pour renforcer mes compétences en **machine learning supervisé**, **feature engineering**, **prétraitement**, et **optimisation d’hyperparamètres**.

Les modèles implémentés comprennent :

* **Régression Logistique (Logistic Regression)**
* **Naive Bayes (GaussianNB)**
* **K-Nearest Neighbors (KNN)**
* **LinearSVC (SVM linéaire)**
* **Voting Classifier (ensemble learning)**

L’objectif final est d’améliorer la **précision**, la **rappel**, le **F1-score** et la **robustesse** des prédictions.


## 🎯 **Objectifs**

* Prétraiter proprement les données médicales (Heart Disease Dataset)
* Construire un pipeline complet : imputation → scaling → split → sauvegarde
* Tester plusieurs modèles de classification supervisée
* Optimiser les modèles avec **GridSearchCV / RandomizedSearchCV**
* Comparer les performances avec plusieurs métriques
* Implémenter un modèle d’**ensemble (VotingClassifier)** pour augmenter la robustesse
* Évaluer la qualité des prédictions via confusion matrix, ROC, PR curves et classification report


## 🧬 **Dataset (Heart Disease)**

Le dataset inclut des variables cliniques telles que :

* âge
* pression sanguine (trestbps)
* cholestérol
* fréquence cardiaque maximale (thalach)
* segment ST (oldpeak)
* nombre de vaisseaux principaux (ca)
* diverses variables catégorielles

La variable cible (`target`) est dérivée du champ `num` et transforme le problème en **classification binaire**.


## 🧹 **Prétraitement des données**

Le prétraitement est effectué dans un pipeline scikit-learn, incluant :

### ✔ Séparation X / y

### ✔ Train/Test Split (stratification pour équilibre des classes)

### ✔ Imputation des valeurs manquantes (SimpleImputer)

### ✔ Normalisation MinMaxScaler

### ✔ Transformation via **ColumnTransformer**

### ✔ Sauvegarde propre en fichiers CSV :

```
X_train_clean.csv
X_test_clean.csv
y_train.csv
y_test.csv
```

Ce pipeline garantit **zéro fuite de données** et une répétabilité parfaite.


## 🤖 **Modèles implémentés**

### 1️⃣ **K-Nearest Neighbors (KNN)**

* Implémentation complète du modèle
* Optimisation des hyperparamètres via **GridSearchCV**
* Test sur différents :

  * n_neighbors
  * weights
  * p (distance)
  * algorithm (ball_tree / kd_tree)

**Métriques analysées** : Accuracy, Precision, Recall, F1.


### 2️⃣ **Naive Bayes (Gaussian Naive Bayes)**

Modèle probabiliste efficace pour les données médicales.
Simple, rapide, performant sur données normalisées.


### 3️⃣ **Régression Logistique**

Un des modèles les plus utilisés en santé pour :

* sa stabilité
* son interprétabilité
* sa capacité à gérer les classes binaires


### 4️⃣ **Support Vector Machine (LinearSVC)**

* Test de plusieurs pénalités
* Optimisation du coefficient C
* Modèle robuste aux données peu séparables


## 🧪 **Ensemble Learning — Voting Classifier**

Un **VotingClassifier (soft voting)** combine plusieurs modèles :

* Logistic Regression
* RandomForestClassifier
* SVC(probability=True)

Objectif :
✔ combiner les forces de chaque modèle
✔ améliorer la stabilité
✔ augmenter la précision sur les cas difficiles


## 📈 **Évaluation**

Les métriques utilisées incluent :

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Classification Report**
* **Confusion Matrix**
* **ROC Curve**
* **Precision-Recall Curve**

Ces visualisations permettent d’analyser les faux positifs/faux négatifs et la qualité du modèle médical.


## 📂 **Structure du repo**

```
heart_disease_classification/
 ├── notebook_preprocessing.ipynb
 ├── knn_model.py
 ├── naive_bayes.py
 ├── logistic_regression.py
 ├── voting_classifier.py
 ├── X_train_clean.csv
 ├── X_test_clean.csv
 ├── y_train.csv
 ├── y_test.csv
 ├── README.md
 └── data/
     └── Heart Disease.csv
```


## 🧠 **Compétences démontrées**

✔ Prétraitement avancé des données
✔ Pipelines scikit-learn professionnels
✔ Feature engineering (création de target binaire)
✔ Entraînement de modèles ML variés
✔ Recherche d’hyperparamètres GridSearchCV/RandomizedSearchCV
✔ Visualisation des performances
✔ Ensemble Learning (VotingClassifier)
✔ Exportation et réutilisation des datasets prétraités


## 🚀 **Améliorations possibles**

* Tester d’autres modèles (XGBoost, Gradient Boosting, RandomForest optimisé)
* Ajouter un SHAP pour l’interprétabilité médicale
* Créer une API (FastAPI / Flask) pour exposer le modèle prédictif
* Construire un Dashboard Streamlit ou Power BI
* Faire une comparaison automatique de tous les modèles dans un tableau final


## 👤 **À propos**

Projet réalisé par **Alex Alkhatib**, passionné par le machine learning, la santé et la modélisation prédictive.


## 📄 Licence
MIT License
Copyright (c) 2025 Alex Alkhatib
