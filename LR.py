#!/usr/bin/python3
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

df = pd.read_csv("data.csv"); 

cv=10
test_frac = 0.2
max_iter = 10000

target = 'good'

X = df.drop([target], axis = 1); y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state = 42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier =  LogisticRegression(max_iter=max_iter)
classifier.fit(X_train, y_train)
training_score = cross_val_score(classifier, X_train, y_train, cv=cv) 
print("For a %1.1f test fraction (%d train & %d test)" %(test_frac, len(X_train), len(X_test)))

print("Unoptimized model")
print("-----------------")
print("Training core = %1.3f" %(training_score.mean()*100))

#7_Cases/3_don.py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]

clf = GridSearchCV(classifier, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
best_clf = clf.fit(X,y)
print(best_clf.best_estimator_)
print(f'Accuracy - : {best_clf.score(X,y):.3f}')
