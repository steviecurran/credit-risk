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

############# MACHINE LEARNING ###############
def ML(n_loops):
    big_array = []
    target = 'good'
 
    for i in range(0,n_loops):
        print("On loop %d of %d" %(i+1,n_loops))
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

        test_frac = 0.2; cv = 10; max_iter = 10000; solver='lbfgs'

        ## DROP TARGET
        X = df.drop([target], axis = 1); y = df[target]
        ## SPLIT DATA
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state = 42)
        ## SCALE FEATURES - TRAIN AND TEST SEPERATELY
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        res=[]
        classifiers = {
            "LR": LogisticRegression(C=10, solver=solver, max_iter = max_iter),
            "KNN": KNeighborsClassifier(n_neighbors=2300), # TAKES A WHILE
            #"SVC": SVC(C=1, gamma=0.1),
            "DTC": DecisionTreeClassifier(max_depth = 10),
        }

        for key, classifier in classifiers.items():
            classifier.fit(X_train, y_train)
            training_score = cross_val_score(classifier, X_train, y_train, cv=cv)
            predictions = classifier.predict(X_test);
            TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
            accuracy = 100*(TP + TN)/(TP + FP + TN + FN)
            
            res.append(round(accuracy,3))

        res2 = np.reshape(res,(-1,3)); 
        big_array.append(res2);

    res3 = np.reshape(big_array,(-1,3));

    data = pd.DataFrame(res3, columns=['LR','kNN','DTC']);

    date = subprocess.check_output('date "+%F-%T"', shell=True).strip()
    # UNIQUE NAME IN CASE WANT TO COMBINE WITH OTHER RUNS
    out = 'ML_loops=%d-%s.csv'  %(n_loops,date)
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
    bins =5 
    
    plt.rcParams.update({'font.size': big})
    plt.figure(figsize=(10,4))
    ax = plt.gca()
    plt.setp(ax.spines.values(),linewidth=2)

    plt.ylabel('Number', size=big); plt.xlabel('Validation accuracy [%]', size=big)

    para = data['LR']
    ax.hist(para, bins=4, color="w", edgecolor='r', linewidth=3);  
    x1, x2 = ax.get_xlim(); y1, y2 = ax.get_ylim()
    #ax.set_ylim([0, 1.5*y2]) # EXTEND y-axis TO FIT TEXT
    y1, y2 = ax.get_ylim()
    x_pos = 77;  # x1;
    y_pos = 0.9*y2; step = (y2-y1)/16
   
    mean = np.mean(para); std = np.std(para); 
    text = "LR:   \u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std)
    plt.text(x_pos,y_pos,text, fontsize = small, c = 'r')

    para = data['kNN']
    ax.hist(para, bins=5, color="w", edgecolor='g', linewidth=3);
    mean = np.mean(para); std = np.std(para); #print(mean,std)
    text = "kNN: \u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std)
    plt.text(x_pos,y_pos-step,text, fontsize = small, c = 'g')

    para = data['DTC'] 
    ax.hist(para, bins=80, color="w", edgecolor='b', linewidth=3, alpha=0.9);
    mean = np.mean(para); std = np.std(para); 
    text = "DTC: \u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std)
    plt.text(x_pos,y_pos-2*step,text, fontsize = small, c = 'b')

    plt.tight_layout()
    plot = "%s" %(out); png = "%s.png" %(plot)
    eps = "convert %s %s.eps; mv %s.eps  media/." % (png, plot,plot); 
    plt.savefig(png); # os.system(eps); # CONVERT BY HAND 
    print("Plot written to", png);
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
