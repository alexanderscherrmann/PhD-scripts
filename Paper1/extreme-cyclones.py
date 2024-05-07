import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec


f = open('/atmosdyn2/ascherrmann/paper/cyc-env-PV/80-per-cyclones.txt','rb')
d = pickle.load(f)
f.close()

df = pd.read_csv('/atmosdyn2/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')

ids = df['ID'].values
lon = df['lon'].values
lat = df['lat'].values
slp = df['minSLP'].values
pv = df['PVsum'].values
mdates = df['date'].values

lons = np.array([])
lats = np.array([])
SLP = np.array([])
intpv = np.array([])
matd = np.array([])

for i in d['ids']:
    lons = np.append(lons,lon[np.where(ids==int(i))[0][0]])
    lats = np.append(lats,lat[np.where(ids==int(i))[0][0]])
    SLP = np.append(SLP,slp[np.where(ids==int(i))[0][0]])
    intpv = np.append(intpv,pv[np.where(ids==int(i))[0][0]])
    matd = np.append(matd,mdates[np.where(ids==int(i))[0][0]])

en = d['envano']
ht = d['htslp']
ex = 0.9
ght = 12
ids = d['ids']
cy = d['cycano']


for j in ['env','cyc']:
    en = d[j + 'ano']
    loc = np.where((en>ex) & (ht>=ght))[0]
    print(j,len(loc),np.percentile(intpv[loc],75),np.mean(intpv[loc]),np.percentile(SLP[loc],25),np.percentile(SLP[loc],75),np.mean(SLP[loc]))

minpltlatc = 15
minpltlonc = -20

maxpltlatc = 60
maxpltlonc = 50

fig=plt.figure(figsize=(6,4))
gs = gridspec.GridSpec(nrows=1, ncols=1)

ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())

ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')



for j,c in zip(['env','cyc'],['k','r']):
    en = d[j + 'ano']
    loc = np.where((en>ex) & (ht>=ght))[0]
    ax.scatter(lons[loc],lats[loc],marker='.',color=c)

fig.savefig('/atmosdyn2/ascherrmann/paper/cyc-env-PV/highly-env.png',dpi=300,bbox_inches='tight')
plt.close('all')

















