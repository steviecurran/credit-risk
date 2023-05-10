#!/usr/bin/python3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

#df = pd.read_csv("loan_data_2007_2014_2.csv");
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

step = 100; stop = 50 # WHICH WILL BE x step

#k_range = list(range(1, 31))
k_range = [k*step for k in range(1,stop)]

k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
    score = 100*(TP + TN)/(TP + FP + TN + FN)
    print(k, score)
    k_scores.append(score)
#k_range, k_scores = np.loadtxt("kNN-results_to_8700.dat", unpack=True) # ONE I PREPARED EARLIER
plt.plot(k_range, k_scores,  c = 'r', lw = 3)
plt.xlabel('Number of nearest neighbours, $k$')
plt.ylabel('kNN validation acccuracy')
plt.tight_layout()
plot = "kNN-results_step=%d_stop=%d" %(step,stop); png = "%s.png" % (plot); 
eps = "convert %s %s.eps; mv %s.eps media/." % (png, plot,plot); 
plt.savefig(png); os.system(eps); plt.show(); print("Plot written to", png);
plt.show()

