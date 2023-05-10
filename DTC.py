#!/usr/bin/python3
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import os 
import sys

from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

df = pd.read_csv("data.csv"); # KEEP TEST SET CONSTANT

cv=10
test_frac = 0.2
target = 'good'

X = df.drop([target], axis = 1); y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state = 42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(6,4))
ax = plt.gca()
plt.setp(ax.spines.values(),linewidth=2)
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    DTC = DecisionTreeClassifier(max_depth=k)
    DTC.fit(X_train, y_train)
    predictions = DTC.predict(X_test)
    TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
    score = 100*(TP + TN)/(TP + FP + TN + FN)
    print(k, score)
    k_scores.append(score)
plt.plot(k_range, k_scores,  c = 'r', lw = 3)
plt.xlabel('Max depth')
plt.ylabel('DTC validation acccuracy')
plt.tight_layout()
plot = "DTC-results"; png = "%s.png" % (plot); 
eps = "convert %s %s.eps; mv %s.eps media/." % (png, plot,plot); 
plt.savefig(png); os.system(eps); plt.show(); print("Plot written to", png);
plt.show()

