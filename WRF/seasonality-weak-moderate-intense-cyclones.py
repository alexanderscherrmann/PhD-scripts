# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
X=''
which = ['weak-' + X + 'cyclones.csv','moderate-' + X + 'cyclones.csv','intense-' + X + 'cyclones.csv']
dp = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
fig,ax = plt.subplots()
for ls,wi in zip(['-','--',':'], which):
    sel = pd.read_csv(dp + wi)
    mo = sel['month']
    freq = np.array([])
    for k in range(1,13):
        freq = np.append(freq,len(np.where(mo==k)[0])/len(mo))
    ax.plot(np.arange(1,13),freq*100,color='k',linestyle=ls)
    
ax.set_xticks(ticks=np.arange(1,13))
ax.set_xticklabels(labels=['J','F','M','A','M','J','J','A','S','O','N','D'])
plt.show()
plt.close('all')
fig,ax = plt.subplots()
for col,wi in zip(['b','k','r'], which):
    sel = pd.read_csv(dp + wi)
    mo = sel['month']
    freq = np.array([])
    for k in range(1,13):
        freq = np.append(freq,len(np.where(mo==k)[0])/len(mo))
    ax.plot(np.arange(1,13),freq*100,color=col)
    
ax.set_xticks(ticks=np.arange(1,13))
ax.set_xticklabels(labels=['J','F','M','A','M','J','J','A','S','O','N','D'])
plt.show()
plt.close('all')
fig,ax = plt.subplots()
for col,wi in zip(['b','k','r'], which):
    sel = pd.read_csv(dp + wi)
    mo = sel['month']
    freq = np.array([])
    for k in range(1,13):
        freq = np.append(freq,len(np.where(mo==k)[0])/len(mo))
    ax.plot(np.arange(1,13),freq*100,color=col)
    
ax.set_xticks(ticks=np.arange(1,13))
ax.set_xticklabels(labels=['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.legend(['weak','moderate','intense'])
fig.savefig(di + 'seasonality-of-weak-moderate-intense-cyclones.png',dpi=300,bbox_inches='tight')
fig.savefig(pi + 'seasonality-of-weak-moderate-intense-cyclones.png',dpi=300,bbox_inches='tight')
%save -r /home/ascherrmann/scripts/WRF/seasonality-weak-moderate-intense-cyclones.py 1-999
