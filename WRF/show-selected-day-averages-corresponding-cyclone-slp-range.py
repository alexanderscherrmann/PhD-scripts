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

f = open('/atmosdyn2/ascherrmann/011-all-ERA5/data/MED-96h-pre-track-deep-over-sea-12h.txt','rb')
tr = pickle.load(f)
f.close()

trID = tr[:,0]


seasons = ['DJF','MAM','JJA','SON']
mn = [np.array([12,1,2]),np.array([3,4,5]),np.array([6,7,8]),np.array([9,10,11])]

fig,ax = plt.subplots()
stat = ax.hist(SLP,bins=39,range=[965,1030],facecolor='grey',edgecolor='k',alpha=1)
ax.axvline(mSLP,color='k')
ax.set_xlabel('SLP [hPa]')
ax.set_ylabel('counts')
ax.set_xlim(960,1030)
fig.savefig(pi + 'SLP-distribution-of-MED-cyclones.png',dpi=300,bbox_inches="tight")
plt.close('all')

for mon,sea in zip(mn,seasons):
    fig,ax = plt.subplots()
    slpsea = np.array([])
    for kk in mon:
        slpsea = np.append(slpsea,SLP[np.where(tmp['months'].values==kk)[0]])

    stat = ax.hist(slpsea,bins=39,range=[965,1030],facecolor='grey',edgecolor='k',alpha=1)
    mseaslp = np.mean(slpsea)
    for c,q,wi in zip(col,range(len(which)),which):
      sel = pd.read_csv(ps + sea + '-' + wi)
      for qq,ll in enumerate([50, 100, 150, 200]):
        selp = sel.iloc[:ll]
        SLPs = selp['minSLP'].values
        ax.axvline(SLPs[-1],color=cc[qq][q])


    ### median and mean are almost identical
    ax.axvline(mseaslp,color='k')
    ax.set_xlabel('SLP [hPa]')
    ax.set_ylim(0,120)
    ax.set_ylabel('counts')
    ax.set_xlim(960,1030)
    
    fig.savefig(pi + 'SLP-distribution-of-MED-cyclones-and-markers-for-intense-weak-cyclones-in-' + sea + '.png',dpi=300,bbox_inches="tight")
    plt.close('all')

    for qq,ll in enumerate([50, 100, 150, 200]):
        fig = plt.figure(figsize=(6,4))
        gs = gridspec.GridSpec(ncols=1, nrows=1)
        ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)

        for c,q,wi in zip(col,range(len(which)),which):
            sel = pd.read_csv(ps + sea + '-' + wi)
            lonstart=np.array([])
            latstart=np.array([])

            selp = sel.iloc[:ll]
            for htmin,i in zip(selp['htSLPmin'].values,selp['ID'].values):
                if htmin>96:
                    htmin=96
                lonstart = np.append(lonstart,tr[np.where(trID==i)[0][0]][-1*(htmin+1)])
                latstart = np.append(latstart,tr[np.where(trID==i)[0][1]][-1*(htmin+1)])

            lon = selp['lon']
            lat = selp['lat']

            ax.scatter(lon,lat,color=c,marker='.',s=5)
            ax.scatter(lonstart,latstart,color=c,marker='x',s=5) 

        lonticks=np.arange(-10,50.1,10).astype(int)
        latticks=np.arange(30,50.1,5).astype(int)
        
        ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
        ax.set_yticks(latticks, crs=ccrs.PlateCarree());
        ax.set_xticklabels(labels=lonticks,fontsize=10)
        ax.set_yticklabels(labels=latticks,fontsize=10)
        
        ax.set_extent([-10,50,30,50],crs=ccrs.PlateCarree())
        
        fig.savefig(pi + 'location-of-%d-MED-cyclones-intense-weak-in-'%ll + sea + '.png',dpi=300,bbox_inches="tight")
        plt.close('all')
