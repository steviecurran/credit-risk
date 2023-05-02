#!/usr/bin/python3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
import collections
import subprocess
from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

df1 = pd.read_csv("loan_data_2007_2014_2.csv"); #
feat = ['good','LP_m','debt'] 

df1 = df1[feat]

############# MACHINE LEARNING ###############
def ML(n_loops):
    big_array = []
    target = 'good'
 
    for i in range(0,n_loops):
        df = df1.sample(frac=1) # randomise
        df[target] = df[target].astype(int)

        good = df[df[target] == 1]; n1 = len(good)
        bad = df[df[target] == 0]; n0 = len(bad)

        if n1 > n0:
            class0 = df.loc[df[target] == 0]
            class1 = df.loc[df[target] == 1][:n0]
        else:
            class0 = df.loc[df[target] == 0][:n1]
            class1 = df.loc[df[target] == 1]
        df = pd.concat([class0, class1])

        df = df.reindex(np.random.permutation(df.index))
        df.reset_index(drop=True, inplace=True)

        test_frac = 0.2; cv = 10; max_iter = 10000

        X = df.drop([target], axis = 1); y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state = 42)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        res=[]
        classifiers = {
            #"LR": LogisticRegression(C=10, solver='newton-cg'),
            #"KNN": KNeighborsClassifier(n_neighbors=15),
            #"SVC": SVC(C=1, gamma=0.1),
            "DTC": DecisionTreeClassifier(max_depth = 10),
        }

        for key, classifier in classifiers.items():
            classifier.fit(X_train, y_train)
            training_score = cross_val_score(classifier, X_train, y_train, cv=cv) 
            res.append(round(training_score.mean(),4))

        res2 = np.reshape(res,(-1,1)); #print(res2)
        big_array.append(res2);

    res3 = np.reshape(big_array,(-1,1));#print(res3)

    data = pd.DataFrame(res3, columns=['DTC']);

    out = 'ML_loops=%d_feat-DTC.csv'  %(n_loops)
    data.to_csv(out, index = False)
    print("Written to %s" %(out))
    
    return out

## HISTOGRAMS
import matplotlib.pyplot as plt
import os 
import sys

def plots(data):
    
    big = 14; small = 12 # text sizes
    data = pd.read_csv(out) 
    bins =10 
    
    plt.rcParams.update({'font.size': big})
    plt.figure(figsize=(6,6))
    ax = plt.gca()
    plt.setp(ax.spines.values(),linewidth=2)

    plt.ylabel('Number', size=big); plt.xlabel('DTC mean scores [%]', size=big)

    para = 100*data['DTC']
    ax.hist(para, bins=bins, color="w", edgecolor='b', linewidth=3);  
    x1, x2 = ax.get_xlim(); y1, y2 = ax.get_ylim()
    #ax.set_ylim([0, 1.5*y2]); y1, y2 = ax.get_ylim()
    x_pos = x1 + (x2-x1)/16; y_pos = 0.9*y2; step = (y2-y1)/16
   
    mean = np.mean(para); std = np.std(para); 
    text = "\u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std)
    plt.text(x_pos,y_pos,text, fontsize = small, c = 'b')

    plt.tight_layout()
    plot = "%s" %(out); png = "%s.png" %(plot)
    eps = "convert %s %s.eps; mv %s.eps  media/." % (png, plot,plot); 
    plt.savefig(png); os.system(eps);  print("Plot written to", png);
    plt.show()

ans = str(input("Run machine learning [could take a while], or straight to plotting histograms? [m/other]: "))

if ans == "m":
    n_loops = int(input("Number of loops [e.g. 100]? "))
    ML(n_loops)
    out = ML(n_loops)

else:
    os.system("ls *loops*csv")
    out = input("Input file?: ")    

plots(out)
