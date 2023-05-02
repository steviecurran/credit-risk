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

df['LP_m_x_debt'] = df['LP_m']*df['debt']*1e-3
para = 'LP_m_x_debt'
print(df[para].describe())
#para = np.log(para)

fact = 1

xmin = min(df[para]); xmax = max(df[para]);
desired_bin_size = 100 # 100000; # int(xmax - xmin)/fact

data = df[para];
min_val = np.min(data); max_val = np.max(data); 
min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
bins = np.linspace(min_boundary, max_boundary, n_bins)

plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(6,6))
ax = plt.gca()
plt.setp(ax.spines.values(),linewidth=2)

text = '%s' %(para)
plt.ylabel('Number', size=14); plt.xlabel("LP_m x debt [$000 months]", size=14)
good = df.loc[df['good'] == 1]
bad = df.loc[df['good'] == 0]; 

ax.hist(good[para], bins=bins, color="w", edgecolor='k', linewidth=0); # TO GET LIMITS, BUT WANT ON TOP 
mean = np.mean(good[para]); std = np.std(good[para])                   # PLOT AGAIN BELOW
ymin, ymax = plt.ylim();

text = "Good: \u03BC = %1.0f, \u03C3 = %1.0f" %(mean,std); xoff = 2.0   
x_pos = xmax-((xmax-xmin)/xoff); y_pos = ymax-((ymax-ymin)/12); y_skip =  ymax-((ymax-ymin)/2);  
plt.text(x_pos,y_pos, text, fontsize = 12, c = 'k', horizontalalignment='left',verticalalignment='top')

ax.hist(bad[para], bins=bins, edgecolor='r', color="w",linewidth=3, alpha= 1);
mean = np.mean(bad[para]); std = np.std(bad[para])

text = "Bad:  \u03BC = %1.0f, \u03C3 = %1.0f" %(mean,std)
plt.text(x_pos,y_pos-y_skip, text, fontsize = 12, c = 'r', horizontalalignment='left',verticalalignment='top')

ax.hist(good[para], bins=bins, color="w", edgecolor='k', linewidth=3, alpha = 0.6)

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

plt.tight_layout()# (pad=0.1)
plot = "2-feat_histo-bin=%d" %(n_bins); png = "%s.png" %(plot);
eps = "convert %s %s.eps; mv %s.eps media/." % (png, plot,plot); 
plt.savefig(png); os.system(eps);print("Plot written to", png);
plt.show()

# To conduct a valid test:Data in each group are normally distributed. # NOT THESE 
