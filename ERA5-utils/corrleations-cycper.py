import pandas as pd
import numpy as np
import xarray as xr

df = pd.read_csv('/home/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')
col = df.columns
df = df.loc[df['ntraj075']>=200]
import matplotlib.pyplot as plt
MONTHS = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
from scipy.stats import pearsonr

NORO = xr.open_dataset('/home/ascherrmann/009-ERA-5/MED/data/NORO')
LON = NORO['lon']
LAT = NORO['lat']

LSM = NORO['OL']

OL = np.array([])
for k in range(len(df['ID'].values)):
    tmlo = df['lon'].values[k]
    tmla = df['lat'].values[k]

    lo = np.where(abs(LON-tmlo)==np.min(abs(LON-tmlo)))[0][0]
    la = np.where(abs(LAT-tmla)==np.min(abs(LAT-tmla)))[0][0]

    OL =np.append(OL,LSM[0,la,lo])


df['LSM'] = OL
cyd = []
data2 = []
slps = []
data = []
envd = []
cyd = []
num = []
trajPV = []
trajano = []
ntraj = []
avdis = []
avpres = []
cycPV= []
envPV = []
htmin = []

for m in MONTHS:
#    ids = np.where(df['mon'].values==m)
    ids = np.where((df['mon'].values==m)&(df['LSM'].values<0.3))[0]
    num.append(len(ids))
    data.append(df['ano'].values[ids])
    data2.append(df['fullano'].values[ids])
    slps.append(df['minSLP'].values[ids])
    cyd.append(df['cycperano'].values[ids])
    envd.append(df['envperano'].values[ids])
    trajPV.append(df['PV075sum'].values[ids]/df['ntraj075'].values[ids])
    trajano.append(df['ano'].values[ids]/df['ntraj075'].values[ids])
    ntraj.append(df['ntraj075'].values[ids])
    avdis.append(df['averagedistance'].values[ids])
    avpres.append(df['averagepressure'].values[ids])
    cycPV.append(df['cycPV'].values[ids]*df['ntraj075'].values[ids])
    envPV.append(df['envPV'].values[ids]*df['ntraj075'].values[ids])
    htmin.append(df['htminSLP'].values[ids])


dat = [data, data2, slps, cyd, envd,trajPV,trajano,ntraj,avdis,avpres,cycPV,envPV,htmin]
sav = ['ano','fullano','slps','cyclonic-ano-per','env-per-ano','PVend','PVend-clim','trajectories','average-distance','average-pressure','cycPV','envPV','htminSLP']
ylab = ['PV anomaly [PVU]','full anomaly [PVU]','minimum SLP [hPa]','cyclonic contribution to anomaly [%]','environmental contribution to anomaly [%]','PVend [PVU]', 'traj anomaly [PVU]','trajectories []','distance to center [km]','initial pressure [hPa]','cyclonic PV [PVU]', 'environmental PV [PVU]','time to minimal SLP [h]']

meanline=dict(linestyle='-',linewidth=1,color='red')
medline=dict(linestyle='-',linewidth=1,color='k')

SLPs = np.array([])
param = np.array([])

for m in range(0,12):
    SLPs = np.append(SLPs,np.mean(slps[m]))
    param =np.append(param,np.mean(envd[m]))

for q,d in enumerate(dat):
    fig,ax = plt.subplots()
    ax.boxplot(d,labels=MONTHS,whis=(10,90),showfliers=False,meanprops=meanline,meanline=True,showmeans=True,medianprops=medline)
    ax.set_xlim(0,13)
    ax.set_xticks(ticks=np.arange(1,13))
    ax.set_xticklabels(labels=MONTHS)
    ax.set_ylabel(ylab[q])
    for k in range(1,13):
        ax.text(0.06 +(k-1)/13,0.975,'%d'%num[k-1],transform=ax.transAxes,fontsize=8)
    fig.savefig('/home/ascherrmann/009-ERA-5/MED/' + sav[q] + '-year-distribution-oversea.png',dpi=300,bbox_inches='tight')
    plt.close('all')

print(pearsonr(param,SLPs))
