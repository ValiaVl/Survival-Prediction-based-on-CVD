# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:19:19 2021

@author: vlaxo
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
# from sklearn.feature_selection import SelectFromModel
from utils import split_sets
# from sklearn import tree
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from utils import plot_precision_recall_curve
# from sklearn.metrics import plot_precision_recall_curve

from features_visualizations import Visual
from feature_and_model_selection import feature_selection, Model_selection
warnings.filterwarnings('ignore')

dataset=pd.read_csv('./heart_failure_clinical_records_dataset.csv')
visual_feat = Visual(dataset)
visual_feat.barplot_cat()

print(dataset.head())
print(dataset.isnull().sum())

X_train, X_test, y_train, y_test = split_sets(dataset)

# =============================================================================
# Feature Selection
# =============================================================================

feature_selection(X_train, y_train, dataset)

# After completing these methods, the algorithms Logistic Regression, SVM, Random Forests και Decision Tree are applied for the total number of features and subsequently for the dominant features ("time", "serum creatinine", "ejection fraction", "serum sodium" και "age")
models_selection=Model_selection(X_train, y_train, X_test, y_test, use_time=True)

# ==============================================================================================
# LOGISTIC REGRESSION IN THE TOTAL NUMBER OF FEATURES AND SUBSEQUENTLY FOR THE DOMINANT FEATURES
# ==============================================================================================
models_selection.logistic_regression()

# ========================================================================================
# RANDOM FOREST IN THE TOTAL NUMBER OF FEATURES AND SUBSEQUENTLY FOR THE DOMINANT FEATURES
# ========================================================================================
models_selection.random_forest()

# =======================================================================================
# DECISION TREE IN THE TOTAL NUMBER OF FEATURES AND SUBSEQUENTLY IN THE DOMINANT FEAUTRES
# =======================================================================================
models_selection.decision_tree()

# Visualize the decision tree that achieved the best performance
models_selection.visualize_best_tree()

# =============================================================================
# SVM IN THE TOTAL NUMBER OF FEATURES
# ============================================================================= 
models_selection.svm()

# =============================================================================
# APPLY THE BEST ALGORITHMS IN THE TEST SET
# ============================================================================= 
results = models_selection.evaluation()
print(results)
