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

df = pd.read_csv('loan_data_2007_2014-Udemy_1.csv',low_memory=False);
outfile = ("loan_data_2007_2014-Udemy_2.csv")

del df['id']
del df['my_good']; df.rename(columns={'ud_good':'good'}, inplace=True) #UDEMY
#del df['ud_good'];df.rename(columns={'my_good':'good'}, inplace=True) #CHECK

print(df)

# ## FUNCTIONS
def woe(data, para ,kind):  # WEIGHT OF EVIDENCE
    data = pd.DataFrame(columns=['n_obs'])
    data['n_obs'] = df.groupby([para]).count()['good'];
    data[para] = data.index
    data = data[[para,'n_obs']]
    data['%n_obs'] = round(100*data['n_obs'] / data['n_obs'].sum(),2)
    data['n_good'] = df.groupby([para]).sum(numeric_only=True)['good']; 
    data['n_bad'] = data['n_obs']-data['n_good']

    data['%good'] = round(100*data['n_good']/data['n_obs'],2)
    data['%bad'] = round(100*data['n_bad'] / data['n_obs'],2)
    data['%n_good'] = round(100*data['n_good'] / data['n_good'].sum(),2)
    data['%n_bad'] = round(100*data['n_bad'] / data['n_bad'].sum(),2)
    data['WoE'] = np.log(data['%n_good'] / data['%n_bad'])

    data['WoE'] = round(np.log(data['%n_good'] / data['%n_bad']),2)
    
    if kind == 'discrete':  # OPTION FOR discrete OR continuous data
         data = data.sort_values(['WoE'])

    data['IV'] = (data['%n_good']/100 - data['%n_bad']/100) * data['WoE']
    temp1 =  data[data['%n_good'] >0]; temp2 = temp1[temp1['%n_bad'] >0] 
    IV_sum = temp2['IV'].sum()
    
    data = data.reset_index(drop=True)

    if IV_sum < 0.02:
        text = "IV < 0.02 - no predictive power"
    elif IV_sum >= 0.02 and IV_sum < 0.1:
        text = "0.02 < IV < 0.1 - weak predictive power"
    elif IV_sum >= 0.1 and IV_sum < 0.3:
        text = "0.1 < IV < 0.3 - medium predictive power"
    elif IV_sum >= 0.3 and IV_sum <= 0.5:   
        text = "0.3 < IV < 0.5 - strong predictive power"
    else:
        text = "IV >  0.5 - something fishy going on"

    #data['d_WoE'] = data['WoE'].diff().abs()
    #data['blah'] = round(data['n_obs']*data['WoE'].diff()/data['WoE'],0); del data['d_WoE']
        
    print(data)
    print("==============================================================================================")
    print("For '%s' feature, sum of IVs = %1.4f: %s" %(para,IV_sum,text))   
    print("==============================================================================================\n\n")
    
    return data #,para,IV_sum #,text

    
def woe_plot(size,rot,para):
    plt.rcParams.update({'font.size': size})
    plt.figure(figsize = (8, 4))
    ax = plt.gca();

    name = df_WoE.columns[0]
    x = np.array(df_WoE.iloc[: , 0].apply(str)) 
    y = df_WoE[para] 
    plt.setp(ax.spines.values(), linewidth=2)
    ax.scatter(x,y, c='k', marker='o',  s=40)
    ax.plot(x,y,c='r',linewidth=3)
    ax.set_xlabel(name); ax.set_ylabel(para)
    plt.xticks(rotation = rot)

    plt.tight_layout() #
    plot = "%s_%s-Udemy" %(str(para),name); png = "%s.png" % (plot); 
    eps = "convert %s %s.eps; mv %s.eps media/." % (png, plot,plot); 
    plt.savefig(png); os.system(eps); print("Plot written to", png);plt.show()

def comb_dummy(para,text):
    arr = []
    new = '%s_%s+' %(para,text[0])

    for i in range (0 ,len(text)):
        col = '%s_%s' %(para,str(text[i]))
        arr.append(col)

    df[new]= df[arr].sum(axis =1)  

    for i in range(0,len(arr)):  # CLEANING UP
        del df[arr[i]]    

print("==================================== GRADE =====================================")
print(df['grade'].unique())

woe(df,'grade','discrete')
df= pd.get_dummies(df,columns= ['grade']); #print(df); #print(df['grade'])

print("==================================== HOME OWNERSHIP =====================================")
df_WoE = woe(df,'home_ownership','discrete')
#woe_plot(14,0,'WoE')

df.rename(columns={'home_ownership':'HO'}, inplace=True);
df = pd.get_dummies(df, columns= ['HO'])
comb_dummy('HO',['RENT', 'OTHER','NONE']);  # FEATURE, MAIN COLUMN, OTHER COLS TO MERGE
comb_dummy('HO',['MORTGAGE','ANY'])
#print(df)

print("==================================== VERIFICATION STATUS =====================================")
df.rename(columns={'verification_status':'VS'}, inplace=True);
df_WoE = woe(df,'VS','discrete')
df = pd.get_dummies(df, columns= ['VS']); #print(df)

print("==================================== PAYMENT PLAN =====================================")
df.rename(columns={'pymnt_plan':'PP'}, inplace=True);
df_WoE = woe(df,'PP','discrete')
del df['PP']

print("==================================== PURPOSE =====================================")
df.rename(columns={'purpose':'P'}, inplace=True);
df_WoE = woe(df,'P','discrete')
#woe_plot(10,90,'WoE')
df = pd.get_dummies(df, columns= ['P'])
comb_dummy('P',['small_business', 'educational','moving','renewable_energy'])
comb_dummy('P',['other','house','medical'])
comb_dummy('P',['debt_consolidation','vacation','wedding'])
comb_dummy('P',['home_improvement','major_purchase'])
comb_dummy('P',['credit_card','car']);

print("==================================== ADDRESS STATE =====================================")
df.rename(columns={'addr_state':'AS'}, inplace=True);
df_WoE = woe(df,'AS','discrete')
# woe_plot(10,90,'WoE')

print(df_WoE.AS.values.tolist())
df = pd.get_dummies(df, columns= ['AS'])
comb_dummy('AS',['NE', 'IA', 'NV'])
comb_dummy('AS',['HI', 'FL','AL'])
comb_dummy('AS',['LA', 'NY', 'NM'])
comb_dummy('AS',['OK', 'NC'])
comb_dummy('AS',['RI', 'SD', 'DE'])
comb_dummy('AS',['AK', 'MS', 'VT'])
comb_dummy('AS',[ 'WY', 'DC', 'ID', 'ME'])

# # ## CHECK
print(df.columns)
cols = ['AS_AR', 'AS_AZ', 'AS_CA', 'AS_CO', 'AS_CT',
       'AS_GA', 'AS_IL', 'AS_IN', 'AS_KS', 'AS_KY', 'AS_MA', 'AS_MD', 'AS_MI', 'AS_MN', 'AS_MO', 'AS_MT', 'AS_NH',
       'AS_NJ', 'AS_OH', 'AS_OR', 'AS_PA', 'AS_SC', 'AS_TN', 'AS_TX', 'AS_UT', 'AS_VA', 'AS_WA', 'AS_WI', 'AS_WV',
       'AS_NE+', 'AS_HI+', 'AS_LA+', 'AS_OK+', 'AS_RI+', 'AS_AK+', 'AS_WY+']
df_AS = df[cols]
for (i,col) in enumerate(cols):
    print(col, df_AS[col].sum())

    
print("==================================== INITIAL LIST STATUS  =====================================")
df.rename(columns={'initial_list_status':'ILS'}, inplace=True);
df_WoE = woe(df,'ILS','discrete')
df = pd.get_dummies(df, columns= ['ILS'])
#print(df)    

df.to_csv(outfile, index=False) # SAVE


