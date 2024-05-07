import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle

import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec

p = '/atmosdyn2/ascherrmann/011-all-ERA5/'
ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'

### find average, moderate cyclone
which = ['weak-cyclones.csv','intense-cyclones.csv']
col = ['blue','red']
cc = [col,['dodgerblue','indianred'],['lightskyblue','tomato'],['cyan','lightcoral']]

df = pd.read_csv(p + 'data/pandas-basic-data-all-deep-over-sea-12h.csv')
MEDend = np.where(df['reg']=='MED')[0][-1] + 1
r = 'MED'
columns = df.columns
tmp = df.loc[df['reg']==r]
SLP = tmp['minSLP'].values
mSLP = np.mean(SLP)
months = tmp['months'].values

seasons = ['DJF','MAM','JJA','SON']
mn = [np.array([12,1,2]),np.array([3,4,5]),np.array([6,7,8]),np.array([9,10,11])]

#fig,ax = plt.subplots()
#stat = ax.hist(SLP,bins=39,range=[965,1030],facecolor='grey',edgecolor='k',alpha=1)
#ax.axvline(mSLP,color='k')
#ax.set_xlabel('SLP [hPa]')
#ax.set_ylabel('counts')
#ax.set_xlim(960,1030)
#fig.savefig(pi + 'SLP-distribution-of-MED-cyclones.png',dpi=300,bbox_inches="tight")
#plt.close('all')


sealist = []
for m,sea in zip(mn,seasons):
    loc = np.where((months==m[0]) | (months==m[1]) | (months==m[2]))[0]
    sealist.append(np.sort(SLP[loc]))

monthslp = []
for mon in np.append(12,np.arange(1,12,1)):
    loc = np.where(months==mon)[0]
    monthslp.append(np.sort(SLP[loc]))


fig,ax = plt.subplots(figsize=(8,6))

lab=['D','J','F','M','A','M','J','J','A','S','O','N']
flier = dict(marker='+',markerfacecolor='grey',markersize=1,linestyle=' ',markeredgecolor='grey')
meanline = dict(linestyle='-',linewidth=1,color='red')
meanline2 = dict(linestyle='-',linewidth=1,color='dodgerblue')
capprops = dict(linestyle='-',linewidth=1,color='grey')
medianprops = dict(linestyle='-',linewidth=1,color='black')
boxprops = dict(linestyle='-',linewidth=1.,color='slategrey')
whiskerprops= dict(linestyle='-',linewidth=1,color='grey')
ax.set_ylabel('min SLP [hPa]')

bp = ax.boxplot(monthslp,whis=(10,90),labels=lab,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops)
ax.set_ylim(990,1015)
fig.savefig(pi + 'monthly-SLP-distribution-of-MED-cyclones.png',dpi=300,bbox_inches="tight")
plt.close('all')


fig,ax = plt.subplots(figsize=(8,6))
lab=seasons
bp = ax.boxplot(sealist,whis=(10,90),labels=lab,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops)
ax.set_ylabel('min SLP [hPa]')
ax.set_ylim(990,1015)
fig.savefig(pi + 'seasonal-SLP-distribution-of-MED-cyclones.png',dpi=300,bbox_inches="tight")
plt.close('all')
