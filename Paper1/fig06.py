import numpy as np
import pickle
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import helper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patch
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import cm
import argparse
import cartopy
import matplotlib.gridspec as gridspec
import pandas as pd

rdis=400
hcyc=0

pload = '/atmosdyn2/ascherrmann/009-ERA-5/MED/ctraj/use/'
pload2 = '/atmosdyn2/ascherrmann/009-ERA-5/MED/traj/use/'
f = open(pload + 'PV-data-dPSP-100-ZB-800-2-%d-correct-distance-noro.txt'%rdis,'rb')
PVdata = pickle.load(f)
f.close()

kickids = np.loadtxt('/atmosdyn2/ascherrmann/009-ERA-5/MED/kick-IDS.txt')

f = open('/atmosdyn2/ascherrmann/009-ERA-5/MED/check-IDS.txt','rb')
getids = pickle.load(f)
f.close()


df = pd.read_csv('/atmosdyn2/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')
### take only the ones with atleast 200 trajectories
thresh = '0.75'
#df = df.loc[df['ntrajgt%s'%thresh]>=200]
df = df.loc[df['ntraj075']>=200]
df = df.loc[df['htminSLP']>=hcyc]

clim = np.loadtxt('/atmosdyn2/ascherrmann/009-ERA-5/MED/clim-avPV.txt')

df2 = pd.DataFrame(columns=['PV','count','avPV'],index=['Year','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
df2['avPV'] = np.append(np.mean(clim),clim)#df2['PV']/df2['count']

dipv = PVdata['dipv']
datadi = PVdata['rawdata']
dit = PVdata['dit']
N = PVdata['noro']
O = PVdata['oro']


both = np.array([])
adv = np.array([])
cyclonic = np.array([])
environmental = np.array([])


SLP = df['minSLP'].values
lon = df['lon'].values
lat = df['lat'].values
ID = df['ID'].values
hourstoSLPmin = df['htminSLP'].values
print(len(ID),len(hourstoSLPmin))
maturedates = df['date'].values

adv = np.array([])
cyc = np.array([])
env = np.array([])
c = 'cyc'
e = 'env'

ac = dict()
pressuredi = dict()

PVstart = np.array([])
PVend = np.array([])
ca = 0
ct = 0
ol = np.array([])
ol2 = np.array([])
pvloc = dict()
cycd = dict()
envd = dict()
envoro = dict()
envnoro = dict()

for h in np.arange(0,49):
    pvloc[h] = np.array([])
    cycd[h] = np.array([])
    envd[h] = np.array([])
    envnoro[h] = np.array([])
    envoro[h] = np.array([])

ncyc = 0
nenv = 0
nadv = 0 

cycano = np.array([])
envano = np.array([])
advano = np.array([])
advano2 = np.array([])

cyper= np.zeros(len(ID))
enper= np.zeros(len(ID))
adper = np.zeros(len(ID))
    
idssave = np.array([])
for ll,k in enumerate(dipv.keys()):
    if np.all(ID!=int(k)):
      continue
    q = np.where(ID==int(k))[0][0]
    if (hourstoSLPmin[q]<(hcyc)):
        continue

    d = k

    OL = PVdata['rawdata'][d]['OL']    
    pre = PVdata['rawdata'][d]['P']
    PV = PVdata['rawdata'][d]['PV']
    i = np.where(PV[:,0]>=0.75)[0]

    pvend = PV[i,0]
    pvstart = PV[i,-1]

    cypv = dipv[d][c][i,0]
    enpv = dipv[d][e][i,0]
    envno = N[d][e][i,0]
    envo = O[d][e][i,0]

    cy = np.mean(cypv)
    
    PVstart = np.append(PVstart,pvstart)
    PVend = np.append(PVend,pvend)

    for h in np.arange(0,49):
        pvloc[h] = np.append(pvloc[h],PV[i,h])
        cycd[h] = np.append(cycd[h],dipv[d][c][i,h])
        envd[h] = np.append(envd[h],dipv[d][e][i,h])
        envoro[h] = np.append(envoro[h],O[d][e][i,h])
        envnoro[h] = np.append(envnoro[h],N[d][e][i,h])


    idssave = np.append(idssave,int(k))


tsc = []
cycbox = []
envbox = []
avlic = np.array([])
tenlic = np.array([])
ninetylic = np.array([])
avlie= np.array([])
tenlie = np.array([])
ninetylie= np.array([])
cy25= np.array([])
cy75= np.array([])
cy50= np.array([])
en25= np.array([])
en75= np.array([])
en50= np.array([])
pvav = np.array([])
pv10 = np.array([])
pv90 = np.array([])

avenoro = np.array([])
enoro10 = np.array([])
enoro25 = np.array([])
enoro75 = np.array([])
enoro90 = np.array([])

avennoro = np.array([])
ennoro10 = np.array([])
ennoro25 = np.array([])
ennoro75 = np.array([])
ennoro90 = np.array([])



for h in np.flip(np.arange(0,49)):
    pvav = np.append(pvav,np.mean(pvloc[h]))
    pv10 = np.append(pv10,np.percentile(pvloc[h],10))
    pv90 = np.append(pv90,np.percentile(pvloc[h],90))
    avlic = np.append(avlic,np.mean(cycd[h]))
    tenlic = np.append(tenlic,np.percentile(np.sort(cycd[h]),10))
    ninetylic = np.append(ninetylic,np.percentile(np.sort(cycd[h]),90))
    avlie = np.append(avlie,np.mean(envd[h]))
    tenlie = np.append(tenlie,np.percentile(np.sort(envd[h]),10))
    ninetylie = np.append(ninetylie,np.percentile(np.sort(envd[h]),90))
    cy25= np.append(cy25,np.percentile(np.sort(cycd[h]),25))
    cy75= np.append(cy75,np.percentile(np.sort(cycd[h]),75))
    cy50= np.append(cy50,np.percentile(np.sort(cycd[h]),50))
    en25= np.append(en25,np.percentile(np.sort(envd[h]),25))
    en75= np.append(en75,np.percentile(np.sort(envd[h]),75))
    en50= np.append(en50,np.percentile(np.sort(envd[h]),50))

    avenoro = np.append(avenoro,np.mean(envoro[h]))
    avennoro =np.append(avennoro,np.mean(envnoro[h]))

    enoro10= np.append(enoro10,np.percentile(np.sort(envoro[h]),10))
    enoro90= np.append(enoro90,np.percentile(np.sort(envoro[h]),90))
    enoro25= np.append(enoro25,np.percentile(np.sort(envoro[h]),25))
    enoro75= np.append(enoro75,np.percentile(np.sort(envoro[h]),75))
    ennoro25= np.append(ennoro25,np.percentile(np.sort(envnoro[h]),25))
    ennoro75= np.append(ennoro75,np.percentile(np.sort(envnoro[h]),75))
    ennoro10= np.append(ennoro10,np.percentile(np.sort(envnoro[h]),10))
    ennoro90= np.append(ennoro90,np.percentile(np.sort(envnoro[h]),90))


fig,ax = plt.subplots()
ax.set_ylabel(r'PV [PVU]')
ax.set_xlabel(r'time to mature stage [h]')
ax.set_ylim(-.75,1.75)
ax.set_xlim(-48,0)
t = np.arange(-48,1)
ax.plot(t,avlic,color='r',linewidth=2)
ax.plot(t,avennoro,color='dodgerblue',linewidth=2)
ax.plot(t,avenoro,color='grey',linewidth=2)

ax.axhline(0,linewidth=1,color='k',zorder=10)
ax.fill_between(t,tenlic,ninetylic,alpha=0.5,color='red')
ax.plot(t,cy25,color='red',linewidth=2.,linestyle='--')
ax.plot(t,cy75,color='red',linewidth=2.,linestyle='--')
ax.plot(t,ennoro25,color='dodgerblue',linewidth=2.,linestyle='--')
ax.plot(t,ennoro75,color='dodgerblue',linewidth=2.,linestyle='--')
ax.fill_between(t,ennoro10,ennoro90,alpha=0.5,color='dodgerblue')
ax.plot(t,enoro25,color='grey',linewidth=2.,linestyle='--')
ax.plot(t,enoro75,color='grey',linewidth=2.,linestyle='--')
ax.fill_between(t,enoro10,enoro90,alpha=0.5,color='grey')

ax.legend(['apvc','apveNO','apveO'],loc='upper left')
ax.set_xticks(ticks=np.arange(-48,1,6))
ax.tick_params(labelright=False,right=True)
ax.set_xticklabels(labels=t[0::6])
ax.text(-0.05, 0.98,'(b)',transform=ax.transAxes,fontsize=12,va='top')
#fig.savefig('/atmosdyn2/ascherrmann/paper/cyc-env-PV/review/env-cyc-PV-lines-mean-t-%d-dis-%d-%s.png'%(hcyc,rdis,thresh),dpi=300,bbox_inches="tight")
fig.savefig('/home/ascherrmann/publications/cyclonic-environmental-pv/fig06b.png',dpi=300,bbox_inches="tight")

plt.close('all')
