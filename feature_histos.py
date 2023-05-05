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

df = pd.read_csv('loan_data_2007_2014_1.csv',low_memory=False);
df = df.select_dtypes(include=np.number); #print(df)

#paras = df.columns; print(paras)
# 'policy_code' # CAUSED A CRASH
#paras = ['acc_now_delinq', 'EM_L', 'ECL_m', 'issued_m', 'LP_m', 'LCP_m', 'debt', 'DTIR']


for (i, para) in enumerate(paras):
 
    fact = 10

    xmin = min(df[para]); xmax = max(df[para]);
    
    if para == "DTIR":
        desired_bin_size = 0.1
    else:
        desired_bin_size = int(xmax - xmin)/fact
     
    
        
    data = df[para]; #print(data)
    min_val = np.min(data); max_val = np.max(data); #print(min_val,max_val)
    min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
    max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
    n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)
    
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(6,4))
    ax = plt.gca()
    plt.setp(ax.spines.values(),linewidth=2)
    
    text = '%s' %(para)
    plt.ylabel('Number', size=14); plt.xlabel(text, size=14)
    good = df.loc[df['good'] == 1]
    bad = df.loc[df['good'] == 0]; #print(len(sus),len(non))
    
    ax.hist(good[para], bins=bins, color="w", edgecolor='grey', linewidth=3);
    mean = np.mean(good[para]); std = np.std(good[para])
    ymin, ymax = plt.ylim();

    if xmax <= 1:
        text = "\u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std); xoff = 2.5    
    elif xmax >1 and xmax <= 10:
        text = "\u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std); xoff = 4.1
    elif xmax >10 and mean <= 100:
        text = "\u03BC = %1.1f, \u03C3 = %1.1f" %(mean,std); xoff = 2.9
    elif xmax >100 and mean <= 1000:
        text = "\u03BC = %1.1f, \u03C3 = %1.0f" %(mean,std);xoff = 2.5  
    elif xmax > 1000 and mean <= 4000: # 10000:
        text = "\u03BC = %1.0f, \u03C3 = %1.0f" %(mean,std); xoff = 3.0
    else:
        text = "\u03BC = %1.0f, \u03C3 = %1.0f" %(mean,std); xoff = 2.5

    x_pos = xmax-((xmax-xmin)/xoff); y_pos = ymax-((ymax-ymin)/12); y_skip =  ymax-((ymax-ymin)/2);  
    plt.text(x_pos,y_pos, text, fontsize = 12, c = 'k', horizontalalignment='left',verticalalignment='top')

    
    ax.hist(bad[para], bins=bins, color="w", edgecolor='r', linewidth=3);
    mean = np.mean(bad[para]); std = np.std(bad[para])

    if xmax < 10:
        text = "\u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std); xoff = 4.1
    elif xmax >10 and mean <= 100:
        text = "\u03BC = %1.1f, \u03C3 = %1.1f" %(mean,std); xoff = 2.9
    elif xmax >100 and mean <= 1000:
        text = "\u03BC = %1.1f, \u03C3 = %1.0f" %(mean,std);xoff = 2.5  
    elif xmax > 1000 and mean <= 10000:
        text = "\u03BC = %1.0f, \u03C3 = %1.0f" %(mean,std); xoff = 3.0
    else:
        text = "\u03BC = %1.0f, \u03C3 = %1.0f" %(mean,std); xoff = 2.5
    
    plt.text(x_pos,y_pos-0.2*y_skip, text, fontsize = 12, c = 'r', horizontalalignment='left',verticalalignment='top')
    
    plt.tight_layout()# (pad=0.1)
    plot = "histo_%s-bin=%d" %(para,fact); png = "%s.png" %(plot);
    eps = "convert %s %s.eps; mv %s.eps media/." % (png, plot,plot); 
    plt.savefig(png); os.system(eps);print("Plot written to", png);
    #plt.show()


