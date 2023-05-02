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

df = pd.read_csv('loan_data_2007_2014_1.csv',low_memory=False); outfile = ("loan_data_2007_2014_2.csv")

del df['id']; del df['loan_status']


## FUNCTIONS
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

    #data = data[[para,'n_obs','%_good','%_n_obs','n_good']]
    data['WoE'] = round(np.log(data['%n_good'] / data['%n_bad']),2)
    #data['WoE_x_n'] = abs(data['WoE']*data['n_obs'])
    
    if kind == 'discrete':  # OPTION FOR discrete OR continuous data
         data = data.sort_values(['WoE'])

    #data = data[data['%n_good'] >0]; data = data[data['%n_bad'] >0] # UMCOMMENT TO GET IVs
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
    plot = "%s_%s" %(str(para),name); png = "%s.png" % (plot); 
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

#df_WoE,para,IV_sum = woe(df,'grade','discrete') # GENERALISED - discrete OR OTHER (continuous)
woe(df,'grade','discrete')
df= pd.get_dummies(df,columns= ['grade']); #print(df); #print(df['grade'])

print("==================================== HOME OWNERSHIP =====================================")
df_WoE = woe(df,'home_ownership','discrete')
#woe_plot(14,0,'WoE')

df.rename(columns={'home_ownership':'HO'}, inplace=True);
df = pd.get_dummies(df, columns= ['HO'])
comb_dummy('HO',['RENT', 'OTHER']);  # FEATURE, MAIN COLUMN, OTHER COLS TO MERGE
comb_dummy('HO',['OWN','NONE','ANY'])
#print(df)
#print(df['HO_RENT+'].sum(),  df['HO_OWN+'].sum()) # CHECK - 188390 41701  AS EXPECTED

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
comb_dummy('P',['small_business', 'moving','renewable_energy'])
comb_dummy('P',['other','house','educational','medical'])
comb_dummy('P',['debt_consolidation','vacation','wedding'])
comb_dummy('P',['home_improvement','major_purchase'])
comb_dummy('P',['credit_card','car']);

print("==================================== ADDRESS STATE =====================================")
df.rename(columns={'addr_state':'AS'}, inplace=True);
df_WoE = woe(df,'AS','discrete')
#woe_plot(10,90,'WoE')

print(df_WoE.AS.values.tolist())
df = pd.get_dummies(df, columns= ['AS'])
comb_dummy('AS',['FL', 'NV', 'HI', 'NE','AL'])
comb_dummy('AS',['LA', 'OK', 'NM'])
comb_dummy('AS',['MO', 'UT', 'TN'])
comb_dummy('AS',['AZ', 'AR', 'SD', 'RI'])
comb_dummy('AS',['KY', 'MN', 'IN', 'MA', 'DE'])
comb_dummy('AS',['GA', 'OR', 'WI'])
comb_dummy('AS',['CT', 'IL'])
comb_dummy('AS',['SC', 'CO', 'AK', 'MT', 'KS'])
comb_dummy('AS',['NH', 'VT', 'MS', 'WV'])
comb_dummy('AS',['WY', 'DC', 'IA', 'ID', 'ME']); #print(df)
## CHECK
print(df.columns)
cols = ['AS_CA', 'AS_MD', 'AS_MI', 'AS_NC','AS_NJ', 'AS_NY', 'AS_OH', 'AS_PA', 'AS_TX', 'AS_VA', 'AS_WA', 'AS_FL+', 'AS_LA+', 'AS_MO+', 'AS_KY+','AS_GA+', 'AS_CT+', 'AS_SC+', 'AS_NH+', 'AS_WY+']
df_AS = df[cols]
for (i,col) in enumerate(cols):
    print(col, df_AS[col].sum())

    
print("==================================== INITIAL LIST STATUS  =====================================")
df.rename(columns={'initial_list_status':'ILS'}, inplace=True);
df_WoE = woe(df,'ILS','discrete')
df = pd.get_dummies(df, columns= ['ILS'])
print(df)    

## CHECKING NO CATEGORICAL VARIABLES LEFT
#temp = df.select_dtypes(exclude=np.number); print(temp.columns) 

#df.to_csv(outfile, index=False) # SAVE

print("----------------------- Important continuous variables -------------------------")
print("====================== Months since last payment (LP_m) ==========================")

#print(df['LP_m'].unique()) # SHOULD BIN
#print(df['LP_m'].describe())

pd.set_option('display.max_row', None)
df_WoE = woe(df,'LP_m','cont')
#woe_plot(10,90,'WoE')

print("====================== Outstanding debt in $000 (debt_k) ==========================")
#df_WoE = woe(df,'debt','cont') # WILL DEFINITELY NEED TO BIN
#print(df['debt'].describe()) # -608.55  TO 35000
df['debt_k'] = round(df['debt']/1000,1); #print(df['debt_k'])
df_WoE = woe(df,'debt_k','cont') #

##############
# print("=========== GLOSSARY ===========") 
# print(" AS = address state")
# print(" debt = loan amount minus the principal received to date")
# print(" ECL  = earliest credit line date")
# print(" ECL_m = months since earliest credit line date")
# print(" EM_L = employment length in years")
# print(" HO = home ownership")
# print(" issued_m = months since loan issued")
# print(" ILS = initial list status")
# print(" LP_m = months since last payment")
# print(" LCP_m = months since last credit pull")
# print(" P = purpose")
# print(" PP = payment plan - DROPPED")
# print(" VS = verification status")
