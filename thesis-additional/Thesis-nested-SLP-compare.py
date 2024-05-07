import sys
sys.path.append('/home/ascherrmann/scripts/')
import wrfsims
import numpy as np
from netCDF4 import Dataset as ds
import matplotlib.pyplot as plt
import wrf
import pickle
import os

SIMS,ATIDS,MEDIDS01,MEDIDS02 = wrfsims.nested_ids()
ntrack = '/atmosdyn2/ascherrmann/scripts/WRF/nested-cyclone-tracking/out/'
trackss = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'
dwrf='/atmosdyn2/ascherrmann/013-WRF-sim/'
sim,at,med=wrfsims.upper_ano_only()
if not os.path.isfile('/atmosdyn2/ascherrmann/013-WRF-sim/nested-vs-normal-domain-MED-slp.txt'):
    track = [trackss,ntrack]
    sims = [sim,SIMS]
    AT = [at,ATIDS]
    MED = [med,MEDIDS02]
    whi = ['normal','nested']
    savedi = dict()
    for sea in ['DJF','MAM']:
     savedi[sea] = dict()
     for tracks,SIM,At,Med,wh in zip(track,sims,AT,MED,whi):
      e5medslp=[]
      for si,a,m in zip(SIM,At,Med):
        if wh=='normal':
#            if si[-4:]=='clim' or 'nested' in si or '0.5' in si or '0.3' in si or 'not' in si or '0.9' in si or '1.1' in si or '1.7' in si or '2.8' in si or sea not in si or 'check' in si or '800' in si:
            if si[-4:]=='clim' or 'nested' in si or 'not' in si or '0.3' in si or '0.5' in si or '2.8' in si or sea not in si or 'check' in si or '800' in si:
                continue
        else:
#            if si[-4:]=='clim'or '0.5' in si or '0.3' in si or 'not' in si or '0.9' in si or '1.1' in si or '1.7' in si or '2.8' in si or sea not in si:
            if si[-4:]=='clim' or 'not' in si or '0.3' in si or '0.5' in si or '2.8' in si or sea not in si:
                continue

        a=np.array(a)
        m=np.array(m)
        if m.size==0:
            continue

        # load
        ic = ds(dwrf + si + '/wrfout_d01_2000-12-01_00:00:00')
        if wh=='normal':
            tra = np.loadtxt(tracks + si + '-new-tracks.txt')
        else:
            tra = np.loadtxt(tracks + si + '-02-new-tracks.txt')
        # store
        t = tra[:,0]
        slp = tra[:,3]
        IDs = tra[:,-1]

        loc = np.where(IDs==2)[0]
        slpmin = np.min(slp[loc])
        print(slpmin,si)
        e5medslp.append(slpmin)
      print(e5medslp)
      savedi[sea][wh] = e5medslp

    f=open('/atmosdyn2/ascherrmann/013-WRF-sim/nested-vs-normal-domain-MED-slp.txt','wb')
    pickle.dump(savedi,f)
    f.close()


f=open('/atmosdyn2/ascherrmann/013-WRF-sim/nested-vs-normal-domain-MED-slp.txt','rb')
data = pickle.load(f)
f.close()
meanbox=[]
labels=[]
for sea in data.keys():
    for wh in data[sea].keys():
        meanbox.append(data[sea][wh])
        labels.append('%s %s'%(sea,wh))

flier = dict(marker='+',markerfacecolor='grey',markersize=1,linestyle=' ',markeredgecolor='grey')
meanline = dict(linestyle='-',linewidth=1,color='red')
meanline2 = dict(linestyle='-',linewidth=1,color='dodgerblue')
capprops = dict(linestyle='-',linewidth=1,color='grey')
medianprops = dict(linestyle='-',linewidth=1,color='black')
boxprops = dict(linestyle='-',linewidth=1.,color='slategrey')
whiskerprops= dict(linestyle='-',linewidth=1,color='grey')

figgg,axxx=plt.subplots(figsize=(8,6))
bp = axxx.boxplot(meanbox,whis=(10,90),labels=labels,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops)
axxx.set_ylabel(r'minimum SLP [hPa]')
axxx.set_xlim(0,5)
axxx.set_ylim(985,1005)
axxx.set_xticklabels(labels=labels)

#plt.xticks(rotation=90)
figgg.savefig('/home/ascherrmann/thesis-images/nested-SLP-vs-normal-domain-box-whis.png',dpi=300,bbox_inches='tight')

plt.close('all')
