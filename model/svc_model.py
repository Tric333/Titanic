#-*- coding: utf-8 -*-

# general lib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns;
import re

from sklearn.preprocessing import *
from sklearn.feature_selection import RFE
from sklearn.decomposition import *

# model
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.linear_model import LinearRegression

svc_model1 = {'classifier': [SVC()], 'preprocessing': [StandardScaler()],
              'classifier__C': [0.1], 'classifier__gamma': [0.1],
              'classifier__probability': [True], 'classifier__kernel': ['poly']}
svc_model2 = {'classifier': [SVC()], 'preprocessing': [KernelPCA(n_components=15)],
              'preprocessing__kernel': ['rbf', 'poly'],
              'classifier__C': np.linspace(0.01, 1, 20), 'classifier__gamma': np.linspace(0.01, 1, 20),
              'classifier__probability': [True], 'classifier__kernel': ['rbf']}

rf_model1 = {'preprocessing':[None],'classifier':[RandomForestClassifier()],
             'classifier__n_estimators':[300,500],'classifier__max_depth':[3,5],'classifier__max_features':['log2'],
             'classifier__min_samples_leaf':[3,5]}
