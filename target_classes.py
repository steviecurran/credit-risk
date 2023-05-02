#!/usr/bin/python3
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mticker
import os 
import sys

from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

df = pd.read_csv('loan_data_2007_2014_1.csv',low_memory=False);
df = df[['loan_status','DTIR','good']]
print(df)

classes = list(df['loan_status'].unique())
print(classes)

de = df[df['loan_status'] == "Default"]; #print(de.describe())
CO = df[df['loan_status'] == "Charged Off"]
VL = df[df['loan_status'] == "Late (31-120 days)"]
ML = df[df['loan_status'] == "Late (16-30 days)"]
IGP = df[df['loan_status'] == "In Grace Period"]
cu = df[df['loan_status'] == "Current"]
DF = df[df['loan_status'] == "Does not meet the credit policy. Status:Fully Paid"]
DC = df[df['loan_status'] == "Does not meet the credit policy. Status:Charged Off"]
FP = df[df['loan_status'] == "Fully Paid"]; #print(FP.describe())

plt.rcParams.update({'font.size':14}) ## COMPARE DISTRIBUTIONS
plt.figure(figsize=(10,4))
ax = plt.gca()
plt.setp(ax.spines.values(),linewidth=2)
plt.ylabel('Numberof loans', size=14); plt.xlabel('Debt to income ratio, DTIR', size=14)

desired_bin_size = 0.01
min_val = np.min(df['DTIR']); max_val = np.max(df['DTIR'])
min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
bins = np.linspace(min_boundary, max_boundary, n_bins)

ax.hist(CO['DTIR'], bins=bins, color="w", edgecolor='r', linewidth=3,alpha = 0.5);

ax.hist(cu['DTIR'], bins=bins, color="w", edgecolor='indigo', linewidth=3,alpha = 0.8);

ax.hist(VL['DTIR'], bins=bins, color="w", edgecolor='y', linewidth=3,alpha = 0.8);

ax.hist(IGP['DTIR'], bins=bins, color="w", edgecolor='b', linewidth=3,alpha = 0.5);

ax.hist(FP['DTIR'], bins=bins, color="w", edgecolor='violet', linewidth=3,alpha = 0.8);

ax.hist(ML['DTIR'], bins=bins, color="w", edgecolor='g', linewidth=3,alpha = 1);

ax.hist(de['DTIR'], bins=bins, color="w", edgecolor='orange', linewidth=3,alpha = 0.5);

ax.hist(DF['DTIR'], bins=bins, color="w", edgecolor='k', linewidth=3,alpha = 0.5);

ax.hist(DC['DTIR'], bins=bins, color="w", edgecolor='grey', linewidth=3,alpha = 1);

xmin, xmax = plt.xlim(); ymin, ymax = plt.ylim(); xoff = 2.8
x_pos = xmax-((xmax-xmin)/xoff); y_pos = ymax-((ymax-ymin)/12); y_skip = ymax-((ymax-ymin)/3);  

ax.set_yscale('log');

text = "Fully Paid: \u03BC = %1.3f, \u03C3 = %1.3f" %(np.mean(FP['DTIR']), np.std(FP['DTIR']))
plt.text(x_pos,10**5.2, text, fontsize = 10, c = 'violet', horizontalalignment='left',verticalalignment='top')


text = "Status:Fully Paid: \u03BC = %1.3f, \u03C3 = %1.3f" %(np.mean(DF['DTIR']), np.std(DF['DTIR']))
plt.text(x_pos,10**4.8, text, fontsize = 10, c = 'k', horizontalalignment='left',verticalalignment='top')

text = "Status:Charged Off: \u03BC = %1.3f, \u03C3 = %1.3f" %(np.mean(DC['DTIR']), np.std(DC['DTIR']))
plt.text(x_pos,10**4.4, text, fontsize = 10, c = 'grey', horizontalalignment='left',verticalalignment='top')

text = "Current: \u03BC = %1.3f, \u03C3 = %1.3f" %(np.mean(cu['DTIR']), np.std(cu['DTIR']))
plt.text(x_pos,10**4.0, text, fontsize = 10, c = 'indigo', horizontalalignment='left',verticalalignment='top')

text = "In Grace Period: \u03BC = %1.3f, \u03C3 = %1.3f" %(np.mean(IGP['DTIR']), np.std(IGP['DTIR']))
plt.text(x_pos,10**3.6, text, fontsize = 10, c = 'b', horizontalalignment='left',verticalalignment='top')

text = "Late (16-30 days): \u03BC = %1.3f, \u03C3 = %1.3f" %(np.mean(ML['DTIR']), np.std(ML['DTIR']))
plt.text(x_pos,10**3.2, text, fontsize = 10, c = 'g', horizontalalignment='left',verticalalignment='top')

text = "Late (31-120 days): \u03BC = %1.3f, \u03C3 = %1.3f" %(np.mean(VL['DTIR']), np.std(VL['DTIR']))
plt.text(x_pos,10**2.8, text, fontsize = 10, c = 'y', horizontalalignment='left',verticalalignment='top')

text = "Default: \u03BC = %1.3f, \u03C3 = %1.3f" %(np.mean(de['DTIR']), np.std(de['DTIR']))
plt.text(x_pos,10**2.4, text, fontsize = 10, c = 'orange', horizontalalignment='left',verticalalignment='top')

text = "Charged Off: \u03BC = %1.3f, \u03C3 = %1.3f" %(np.mean(CO['DTIR']), np.std(CO['DTIR']))
plt.text(x_pos,10**2.0, text, fontsize = 10, c = 'r', horizontalalignment='left',verticalalignment='top')

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
plot = 'target_classes'; png = "%s.png" %(plot);
eps = "convert %s %s.eps; mv %s.eps media/." % (png, plot,plot); 
plt.savefig(png); os.system(eps);print("Plot written to", png);
plt.show()
