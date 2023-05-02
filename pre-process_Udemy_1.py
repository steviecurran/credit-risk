#!/usr/bin/python3
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mticker
import os 
import sys
import scipy

from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

df = pd.read_csv('loan_data_2007_2014.csv',low_memory=False);
print(df);#print(df.info())

## MISSING VALUES 
def max_missing(data,most):
    cols = data.columns
    miss = []
    for (i,col) in enumerate(cols):
        if data[col].isnull().sum() > most:
            print("%s has %d missing values" %(col,data[col].isnull().sum()))
            miss.append(col)

    for (i,col) in enumerate(miss):   # REMOVING
        del data[col]
max_missing(df,200000) # TO START WITH GET RID OF THOSE WITH > 200,000 MISSING VALUES
print("--------------------------")
del df['Unnamed: 0'];
#del df['id'];  # KEEP FOR IMPUTATION
del df['member_id'] 
print(df)

## EMPLOYMENT LENGTH
print(df['emp_length'].unique())
df['EM_L'] = df['emp_length'].str.replace('\+ years', '', regex=True)
df['EM_L'] = df['EM_L'].str.replace('< 1 year', str(0))
df['EM_L'] = df['EM_L'].str.replace('n/a',  str(0))
df['EM_L'] = df['EM_L'].str.replace(' years', '')
df['EM_L'] = df['EM_L'].str.replace(' year', '')
del df['emp_length']
df['EM_L'] = df['EM_L'].astype(float)

## TERM OF LOAN
print(df['term'].unique())
df['term'] = df['term'].str.replace('months', '');
df['term'] = pd.to_numeric(df['term']);

print(df['EM_L'].unique()); print(df['term'].unique())

## EARLIEST CREDIT LINE
#print(df.columns)
from datetime import date
from datetime import datetime

def date_fix(orig,para,newday,newmonth,newest):  # do as function so can be re-used
    global df

    print(df[orig])
    df[para] = pd.to_datetime(df[orig], format = '%b-%y')
    print(df[para]); print(df[para].describe())

    today = np.datetime64(datetime.today().date()); print(today)
    temp = df[df[para] < today]; print(temp[para].describe())
  
    latest = pd.to_datetime(newest); 
    #df['earliest_cr_line'].to_csv("check.dat", index=False) # CHECK VIA SHELL (grep) - CHECKS OUT
    
    df['DYS'] = latest - df[para] # days
    df['MNTHS'] = df['DYS']/np.timedelta64(1, 'M'); print(df['MNTHS'].describe())
    
    df['Other_MNTHS'] = df['MNTHS'] ## OTHERS' METHOD - subsituting the negative values...
    df.loc[df['MNTHS'] < 0, 'Other_MNTHS'] = max(df['MNTHS'])
    print(df['Other_MNTHS'].describe())
    
    df['tmp'] = df[para].astype(str);  ## MY METHOD
    df['year'] = df['tmp'].str.split('-').str[0]; #
    df['month'] = df['tmp'].str.split('-').str[1];
    df['day'] = df['tmp'].str.split('-').str[2]; # print(df); print(type(df['year']))

    df = df[df['year'] != 'NaT']; #print(df)
    df['year'] = df['year'].astype(int)

    df.loc[df['MNTHS'] < 0, 'year'] = df['year']-100
    df[para] = pd.to_datetime(df[['year','month','day']])
    df[newday] = latest - df[para] # days
    df[newmonth] = df[newday]/np.timedelta64(1, 'M');
    print(df[newmonth].describe())

    plt.rcParams.update({'font.size':14}) ## COMPARE DISTRIBUTIONS
    plt.figure(figsize=(10,4))
    ax = plt.gca()
    plt.setp(ax.spines.values(),linewidth=2)
    plt.ylabel('Number', size=14); plt.xlabel('Months to %s since %s' %(newest,para), size=14)

    desired_bin_size = 24
    min_val = np.min(df[newmonth]); max_val = np.max(df[newmonth])
    min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
    max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
    n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)

    ax.hist(df[newmonth], bins=bins, color="w", edgecolor='b', linewidth=3, alpha =1);
    mean = np.mean(df[newmonth]); std = np.std(df[newmonth])
    xmin, xmax = plt.xlim(); ymin, ymax = plt.ylim(); xoff = 2.8
    text = "This method:     \u03BC = %1.0f, \u03C3 = %1.0f" %(mean,std)
    x_pos = xmax-((xmax-xmin)/xoff); y_pos = ymax-((ymax-ymin)/12); y_skip =  ymax-((ymax-ymin)/3);  
    plt.text(x_pos,y_pos, text, fontsize = 12, c = 'b', horizontalalignment='left',verticalalignment='top') 

    ax.hist(df['Other_MNTHS'], bins=bins, color="w", edgecolor='r', linewidth=3,alpha = 0.5);
    mean = np.mean(df['Other_MNTHS']); std = np.std(df['Other_MNTHS'])
    text = "Udemy assumption: \u03BC = %1.0f, \u03C3 = %1.0f" %(mean,std)
    plt.text(x_pos,y_pos-y_skip, text, fontsize = 12, c = 'r', horizontalalignment='left',verticalalignment='top') 

    ax.set_yscale('log');
    def update_ticks(z, pos):
        if z ==1:
            return '1 '
        elif z >1 and z <1000:
            return '%d' %(z)
        elif z < 1 and z > 0.001:
            return z
        else:
            return  '10$^{%1.0f}$' %(np.log10(z)) 

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
    plt.tight_layout()
    plot = '%s_histos-%s' %(para,newest); png = "%s.png" %(plot);
    eps = "convert %s %s.eps; mv %s.eps media/." % (png, plot,plot); 
    #plt.savefig(png); os.system(eps);print("Plot written to", png);plt.show()
    print(df[para].describe())
    del df[orig]

date_fix('earliest_cr_line','ECL','temp_d','ECL_m','2011-11-01')    

# ## ISSUE DATE - repeat the above
date_fix('issue_d','issued','temp_d','issued_m','2014-12-01')
date_fix('last_pymnt_d', 'date','temp_d','LP_m','2016-01-01')
date_fix('last_credit_pull_d', 'date','temp_d','LCP_m','2016-01-01')

del df['ECL']; del df['issued'];  # NO LONGER NEED
del df['DYS']; del df['MNTHS']; del df['tmp']; del df['temp_d']; del df['date']
del df['Other_MNTHS']; del df['year']; del df['month']; del df['day']
 
## FURTHER TRIMMING
del df['url']
print(df['acc_now_delinq'].unique()) # [0. 1. 2. 3. 5. 4.
del df['application_type']; del df['zip_code']; del df['sub_grade']

max_missing(df,30000)

def uniques(para):
    print(df[para].unique())
    print("%s has %d unique values" %(para, len(df[para].unique())))
del df['emp_title']
del df['title']
uniques('EM_L')

## IMPUTE REMAINING MISSING VALUES - THIS IS DEFINTELY MESSING WITH LATER CLASSIFICATION
## ISOLATE FROM CATEGORICAL VALUES, WILL ALSO REMOVE TARGET, WHICH WE WANT
## LREADY DOING - IS IT THE RECOMB?
df_orig = df.copy()
cols = ['revol_util','collections_12_mths_ex_med','EM_L']
print("------------ BEFORE IMPUTATION -------------- ")
for (i,col) in enumerate(cols):
    print("For %s, n = %d, mean = %1.3f +/- %1.3f" %(col,df[col].count(), np.mean(df[col]),np.std(df[col])))

df_num = df.select_dtypes(include=np.number)  # NUMERICAL FIELDS ONLY
df_rest=df[['id','grade','home_ownership', 'verification_status', 'loan_status', 'pymnt_plan', 'purpose', 'addr_state', 'initial_list_status']]

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer      
imp = IterativeImputer(max_iter=1000, random_state=0)

imp_array = imp.fit_transform(df_num) # OUTPUT IS ARRAY WHICH HAS TO BE PUT BACK INTO DATA_FRAME
df_imp = pd.DataFrame(imp_array) 
df_imp.columns = df_num.columns

print("------------ AFTER IMPUTATION -------------- ")
for (i,col) in enumerate(cols):
    print("For %s, n = %d, mean = %1.3f +/- %1.3f" %(col,df_imp[col].count(), np.mean(df_imp[col]),np.std(df_imp[col])))

df = pd.merge(df_imp, df_rest, on = ['id'], how = 'outer');

##### MY CLASSIFICATION #########
df['debt'] = df['loan_amnt'] - df['total_rec_prncp']; #print(df['debt'])
df['DTIR'] = df['debt']/df['annual_inc'] 

arr1 = []; good = []; bad = []
cols = df['loan_status'].unique();# print(cols)
for (i,col) in enumerate(cols):
    temp = df[df['loan_status'] ==col]
    mean = np.mean(temp['DTIR']); sd = np.std(temp['DTIR']); n = temp['DTIR'].count()
    arr1.append(col)
    arr1.append(n)
    arr1.append(round(mean,4))
    arr1.append(round(sd/(n**0.5),4))
    if mean < 0.15:  # FOR MAPPING VALUES
        good.append(col)
    else:
        bad.append(col)
    
arr2 = np.reshape(arr1,(-1,4)); #print(arr2)
df1 = pd.DataFrame(arr2, columns=['loan_status','number','mean DTIR','standard error'])
print(df1.sort_values(by=['mean DTIR']))

df['my_good'] = df['loan_status'].map({good[i]:1 for i in range(0,len(good))})
df['my_good']= df['my_good'].fillna(0)
df['my_good'] = df['my_good'].astype(int)

#print(df[['loan_status','my_good']])
good = df[df['my_good']==1]; print("Giving %d good loans (good = 1)" %(len(good)))
bad = df[df['my_good']==0]; print(" and %d bad loans (good = 0)" %(len(bad)))    

print("====== my_good = 1 ======")
print(good['loan_status'].unique())
print("====== my_good = 0 ======")
print(bad['loan_status'].unique())
#print("====== Stats ======")      
print(df['my_good'].describe())

##### UDEMY CLASSIFICATION #########
df['ud_good'] = np.where(df['loan_status'].isin(['Charged Off', 'Default',
                                                  'Does not meet the credit policy. Status:Charged Off',
                                                  'Late (31-120 days)']), 0, 1)

good = df[df['ud_good']==1]; print("Giving %d good loans (good = 1)" %(len(good)))
bad = df[df['ud_good']==0]; print(" and %d bad loans (good = 0)" %(len(bad)))    
print("====== Udemy good = 1 ======")
print(good['loan_status'].unique())
print("====== Udemy _good = 0 ======")
print(bad['loan_status'].unique())
#print("====== Stats ======")      
print(df['ud_good'].describe())
######################################################

del df['loan_status']

df.to_csv("loan_data_2007_2014-Udemy_1.csv", index=False) # SAVE

