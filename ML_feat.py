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
import pickle
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os 
import sys
from sklearn.metrics import roc_curve, roc_auc_score

from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

df = pd.read_csv("loan_data_2007_2014_2.csv"); #print(df) 

#feat = ['good','total_pymnt_inv','funded_amnt_inv','out_prncp_inv','total_rec_prncp','out_prncp','total_rec_int','recoveries','loan_amnt','installment','LP_m','debt','total_pymnt','last_pymnt_amnt','funded_amnt','total_rec_late_fee'] # IMPORTANCE > 0.01

#feat = ['good','total_pymnt_inv','funded_amnt_inv','out_prncp_inv','total_rec_prncp','out_prncp','total_rec_int','recoveries','loan_amnt','installment'] # > 0.05

feat = ['good','total_pymnt_inv','funded_amnt_inv','out_prncp_inv','total_rec_prncp','out_prncp'] # > 0.1

#feat = ['good','total_pymnt_inv','funded_amnt_inv','out_prncp_inv'] # >~ 0.2

#df = df[feat] ; print(df.head())

############# MACHINE LEARNING ###############
df = df.sample(frac=1) 
target = 'good'

good = df[df[target] == 1]; n1 = len(good)
bad = df[df[target] == 0]; n0 = len(bad)

if n1 > n0:
    class0 = df.loc[df[target] == 0]
    class1 = df.loc[df[target] == 1][:n0]
else:
    class0 = df.loc[df[target] == 0][:n1]
    class1 = df.loc[df[target] == 1]
print("Sample sizes now", len(class0),len(class1))
df = pd.concat([class0, class1]); #print(df)

df = df.reindex(np.random.permutation(df.index))
df.reset_index(drop=True, inplace=True)

test_frac = 0.2
max_iter = 10000
cv = 10
solver='lbfgs'
## DROP TARGET
X = df.drop([target], axis = 1); y = df[target]
## SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state = 42)
## SCALE FEATURES - TRAIN AND TEST SEPERATELY
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classifiers = {
    "LR": LogisticRegression(C=10, solver=solver, max_iter = max_iter),
    #"DTC": DecisionTreeClassifier(max_depth = 10), 
    #"KNN": KNeighborsClassifier(n_neighbors=2000), # JUST FOR A LOOK
}

for key, classifier in classifiers.items():

    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=cv) 
    print("For a %1.1f test fraction (%d train & %d test) %s score = %1.3f" %(test_frac, len(X_train), len(X_test), key, training_score.mean()*100))
    predictions = classifier.predict(X_test); #print('Testing\n', confusion_matrix(predictions, y_test))
    TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
    print('Validation accuracy of %s is %1.2f percent' %(key, 100*(TP + TN)/(TP + FP + TN + FN)))
    
    ####### KS STUFF  #################
    y_hat_test = classifier.predict(X_test)
    y_hat_test_proba = classifier.predict_proba(X_test); #print(y_hat_test_proba)
    y_hat_test_proba = y_hat_test_proba[: ][: , 1]; #print(y_hat_test_proba)

    temp = y_test
    temp.reset_index(drop = True, inplace = True)
    df_probs = pd.concat([temp, pd.DataFrame(y_hat_test_proba)], axis = 1)
    df_probs.columns = ['y_test', 'y_hat_test_proba']
    df_probs.index = y_test.index; #print(df_probs)
    df_probs['y_hat_test'] = np.where(df_probs['y_hat_test_proba'] > 0.5, 1, 0)
    
    df_probs = df_probs.sort_values('y_hat_test_proba')
    df_probs = df_probs.reset_index()
    df_probs['Cumulative N Population'] = df_probs.index + 1
    df_probs['Cumulative N Good'] = df_probs['y_test'].cumsum()
    df_probs['Cumulative N Bad'] = df_probs['Cumulative N Population'] - df_probs['y_test'].cumsum()

    df_probs['Cumulative Perc Population']=df_probs['Cumulative N Population']/(df_probs.shape[0])
    df_probs['Cumulative Perc Good']=df_probs['Cumulative N Good']/df_probs['y_test'].sum()
    df_probs['Cumulative Perc Bad']=df_probs['Cumulative N Bad']/(df_probs.shape[0] - df_probs['y_test'].sum())
    print(df_probs);
    KS = max(df_probs['Cumulative Perc Bad'] - df_probs['Cumulative Perc Good'])
    print("KS = %1.3f" %(KS))

     ##### CURVEs ############
    #fpr, tpr, _ = metrics.roc_curve(y_test, y_hat_test)  # NOT REALLY A CURVE???
    #auc = metrics.roc_auc_score(y_test, y_hat_test); gini = 2*auc-1
    #print("    AUC = %1.4f and Gini = %1.2f%%" %(auc,100*gini))
    fpr, tpr, thresholds = roc_curve(df_probs['y_test'],df_probs['y_hat_test_proba'])
    auc = roc_auc_score(df_probs['y_test'], df_probs['y_hat_test_proba']); gini = 2*auc-1
    print("    AUC = %1.4f and Gini = %1.2f%%" %(auc,100*gini))
    #print(fpr, tpr, thresholds)
    
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(6,4))
    ax = plt.gca()
    plt.setp(ax.spines.values(),linewidth=2)

    plt.plot(fpr, tpr, linestyle = '-',  lw =3, color = 'b', label="ROC")
    plt.plot(fpr, fpr, linestyle = '--',  lw =3, color = 'r', label="Benchmark")
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.text(0.65,0.05, "%s: AUC = %1.3f" %(key,auc), color = 'k', fontsize = 12)
    plt.tight_layout()
    plot = "%s-AUC_plot" %(key); png = "%s.png" % (plot);
    eps = "convert %s %s.eps; mv %s.eps media/." % (png, plot,plot); 
    plt.savefig(png); os.system(eps); print("Plot written to", png);plt.show();
    plt.clf(); plt.cla(); plt.close()    

    ######## KS PLOT ##############

    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(6,4))
    ax = plt.gca()
    plt.setp(ax.spines.values(),linewidth=2)

    plt.plot(df_probs['y_hat_test_proba'],df_probs['Cumulative Perc Bad'],lw =3,color = 'r',label="Bad")
    plt.plot(df_probs['y_hat_test_proba'],df_probs['Cumulative Perc Good'],lw =3,color = 'b',label="Good")

    plt.xlabel('Estimated probability of good = 1');plt.ylabel('Cumulative %')

    plt.text(0.05,0.9, "%s: KS = %1.3f" %(key,KS), color = 'k', fontsize = 12)
    plt.tight_layout()
    plot = "%s-KS_plot" %(key); png = "%s.png" % (plot);
    eps = "convert %s %s.eps; mv %s.eps media/." % (png, plot,plot); 
    plt.savefig(png); os.system(eps); print("Plot written to", png);plt.show()
    plt.clf(); plt.cla(); plt.close()   
    
    #FEATURE IMPORTANCE #########
    # from sklearn.inspection import permutation_importance
    # pd.set_option('display.max_rows', None)
    
    # results = permutation_importance(classifier, X_train, y_train, scoring='accuracy')
    # importance = results.importances_mean
    # features = df.drop(target,axis = 1) 
    # features = features.columns.tolist()
    # df1 = pd.DataFrame(features, columns=['Feature']); 
    # df2 = pd.DataFrame(importance, columns=['Importance'])
    # df1['Importance'] = df2.Importance  # adding to df1
    # print(df1.sort_values(by=['Importance'], ascending=False)) 

    
