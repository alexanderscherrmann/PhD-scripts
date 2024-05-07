import sys
sys.path.append('/home/ascherrmann/scripts/')
import wrfsims
import numpy as np
from netCDF4 import Dataset as ds
import matplotlib.pyplot as plt
import wrf
import pickle
import os

sim,at,med=wrfsims.cesm_ids()

tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'
ics='/atmosdyn2/ascherrmann/013-WRF-sim/ics/'
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
sea='DJF'

colors=['k','blue','yellow','orange','red']
x0,y0,x1,y1=70,30,181,101

ofsetfac=[0,0.5,1,2]
xof=[0,-8,8,0,0]
yof=[0,0,0,-8,8]
names = ['-0-km','west','east','south','north']
km=['-0-km','200','400','800']
period=['ERA5','2010','2040','2070','2100']
refx=dict()
refy=dict()

if os.path.isfile('/atmosdyn2/ascherrmann/015-CESM-WRF/CESM-slp-PV-data-no-weak-experiments.txt'):
    f = open('/atmosdyn2/ascherrmann/015-CESM-WRF/CESM-slp-PV-data.txt','rb')
    save = pickle.load(f)    
    f.close()

    pvdi=save['pv']
    atslpdi=save['at']
    medslpdi=save['med']

f=open('/atmosdyn2/ascherrmann/015-CESM-WRF/paper-ERA5-slp-pv-data-no-weak-experiments.txt','rb')
e5dat = pickle.load(f)
f.close()


flier = dict(marker='+',markerfacecolor='grey',markersize=1,linestyle=' ',markeredgecolor='grey')
meanline = dict(linestyle='-',linewidth=1,color='red')
meanline2 = dict(linestyle='-',linewidth=1,color='dodgerblue')
capprops = dict(linestyle='-',linewidth=1,color='grey')
medianprops = dict(linestyle='-',linewidth=1,color='black')
boxprops = dict(linestyle='-',linewidth=1.,color='slategrey')
whiskerprops= dict(linestyle='-',linewidth=1,color='grey')


#meanbox.append(jetmedslp)

### MED
labels=np.array(['ERA5 42y','E2010','C2010'])#,'CESM SC','CESM MC','CESM EC'])#,'CESM EC'])

fig,axes=plt.subplots(figsize=(9,6),nrows=2,ncols=2,sharex=True)

meanbox=[e5dat['med']]
for perio in period[:2]:
    meanbox.append(medslpdi[perio])

ax = axes[1,0]
bp = ax.boxplot(meanbox,whis=(10,90),labels=labels,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops)

ax.set_ylabel(r'minimum SLP [hPa]')
#xx.set_xlim(0,8)
ax.set_xlim(0,4)
ax.set_ylim(990,1015)
ax.set_yticklabels(labels=np.arange(990,1015,5))
ax.text(0.01,0.93,'(c) Mediterranean cyclones',transform=ax.transAxes)
#ax.set_xticklabels(labels=labels)

#plt.xticks(rotation=90)
#fig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/images/SLP-box-whiskers.png',dpi=300,bbox_inches='tight')
#plt.close('all')


## atlantic
meanbox=[e5dat['at']]
for perio in period[:2]:
    meanbox.append(atslpdi[perio])

#meanbox.append(jetatslp)

#fig,axes=plt.subplots(figsize=(9,6),sharex=True,sharey=True)
ax=axes[0,0]
bp = ax.boxplot(meanbox,whis=(10,90),labels=labels,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops)
ax.set_ylabel(r'minimum SLP [hPa]')
ax.set_xlim(0,4)
ax.set_ylim(960,1010)
ax.set_yticklabels(labels=np.arange(960,1011,10))
ax.text(0.01,0.93,'(a) Atlantic cyclones',transform=ax.transAxes)
#ax.set_xticklabels(labels=labels)

#plt.xticks(rotation=90)
#fig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/images/SLP-Atlantic-box-whiskers.png',dpi=300,bbox_inches='tight')


f = open('/atmosdyn2/ascherrmann/015-CESM-WRF/CESM-delta-SLP-track-no-weak-experiments.txt','rb')
savec = pickle.load(f)
f.close()

f = open('/atmosdyn2/ascherrmann/015-CESM-WRF/delta-SLP-track-ERA5-paper-no-weak-experiments.txt','rb')
savee5= pickle.load(f)
f.close()

ax=axes[1,1]


tmpn = np.array([])
for k in savee5['med']['DJF']:
    tmpn = np.append(tmpn,k)
meanbox=[tmpn]

#print(meanbox)
for perio in list(savec['med'].keys())[:2]:
    tmpn = np.array([])
    for k in savec['med'][perio]:
        tmpn = np.append(tmpn,k)
    meanbox.append(tmpn)

bp = ax.boxplot(meanbox,whis=(10,90),labels=labels,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops)
ax.set_ylabel(r'min SLP - clim. SLP [hPa]')
ax.set_xlim(0,4)
ax.set_ylim(-25,0)
ax.text(0.01,0.93,'(d) Mediterranean cyclones',transform=ax.transAxes)
ax.set_yticklabels(labels=np.arange(-25,0,5))
ax=axes[0,1]

tmpn = np.array([])
for k in savee5['at']['DJF']:
    tmpn = np.append(tmpn,k)
meanbox=[tmpn]

#print(meanbox)
for perio in list(savec['at'].keys())[:2]:
    tmpn = np.array([])
    for k in savec['at'][perio]:
        tmpn = np.append(tmpn,k)
    meanbox.append(tmpn)

bp = ax.boxplot(meanbox,whis=(10,90),labels=labels,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops)
ax.set_ylabel(r'min SLP - clim. SLP [hPa]')
ax.set_xlim(0,4)
ax.set_ylim(-35,0)
ax.text(0.01,0.93,'(b) Atlantic cyclones',transform=ax.transAxes)
ax.set_yticklabels(labels=np.arange(-35,1,5))
plt.subplots_adjust(hspace=0)

fig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/images/box-plots-ERA5-40y-PD-CESM-PD-no-weak-experiments.png',dpi=300,bbox_inches='tight')
plt.close('all')

