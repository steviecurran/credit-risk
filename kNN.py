#!/usr/bin/python3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import matplotlib
import os 
import sys

from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

# df = pd.read_csv("loan_data_2007_2014_2.csv");

# df = df.sample(frac=1) 
# target = 'good'

# good = df[df[target] == 1]; n1 = len(good)
# bad = df[df[target] == 0]; n0 = len(bad)

# if n1 > n0:
#     class0 = df.loc[df[target] == 0]
#     class1 = df.loc[df[target] == 1][:n0]
# else:
#     class0 = df.loc[df[target] == 0][:n1]
#     class1 = df.loc[df[target] == 1]
# print("Sample sizes now", len(class0),len(class1))
# df = pd.concat([class0, class1]); #print(df)

# df = df.reindex(np.random.permutation(df.index))
# df.reset_index(drop=True, inplace=True)

# cv=10
# test_frac = 0.2

# X = df.drop([target], axis = 1); y = df[target]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state = 42)

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(6,4))
ax = plt.gca()
plt.setp(ax.spines.values(),linewidth=2)

# #k_range = list(range(1, 301))
# #k_range = [k*10 for k in range(1, 50)]
# k_range = [k*100 for k in range(1, 500)]
# k_scores = []
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
#     print(k, scores.mean())
#     k_scores.append(scores.mean())
# print(k_scores)
#plt.plot(k_range, k_scores,  c = 'r', lw = 3)

k_range, k_scores = np.loadtxt("kNN-results_to_8700.dat", unpack=True) # ONE I PREPARED EARLIER
plt.plot(k_range, k_scores,  c = 'r', lw = 3)

plt.xlabel('Number of nearest neighbours, $k$')
plt.ylabel('kNN cross-validation acccuracy')
plt.tight_layout()
plot = "kNN-results"; png = "%s.png" % (plot); 
eps = "convert %s %s.eps; mv %s.eps media/." % (png, plot,plot); 
plt.savefig(png); os.system(eps); plt.show(); print("Plot written to", png);
plt.show()

