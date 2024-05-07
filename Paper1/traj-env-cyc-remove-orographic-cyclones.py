rdis=400
hcyc=0
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
import xarray as xr

pload = '/atmosdyn2/ascherrmann/009-ERA-5/MED/ctraj/use/'
pload2 = '/atmosdyn2/ascherrmann/009-ERA-5/MED/traj/use/'
f = open(pload + 'PV-data-dPSP-100-ZB-800-2-%d-correct-distance-noro.txt'%rdis,'rb')
PVdata = pickle.load(f)
f.close()
data = PVdata
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
    
df = pd.read_csv('/atmosdyn2/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')
#thresh='1.5PVU'
thresh='075'
df = df.loc[df['ntraj%s'%thresh]>=200]
ID = df['ID'].values

NORO = xr.open_dataset('/atmosdyn2/ascherrmann/009-ERA-5/MED/data/NORO')
LON  = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)
OL = NORO['OL'][0]

clim = np.loadtxt('/atmosdyn2/ascherrmann/009-ERA-5/MED/clim-avPV.txt')
df2['avPV'] = np.append(np.mean(clim),clim)

df2 = pd.DataFrame(columns=['avPV'],index=['Year','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
df2['avPV'] = np.append(np.mean(clim),clim)
poroid = np.array([])
oro = data['oro']
for k in dipv.keys():
    if np.all(ID!=int(k)):
        continue

    i = np.where(ID==int(k))[0][0]
    mon = df['mon'].values[i]

    idp = np.where((datadi[k]['PV'][:,0]>=0.75)&(datadi[k]['P'][:,0]>=700))[0]

    if np.mean(oro[k]['env'][idp,0])<0.15:
        continue

    if np.mean(dipv[k]['env'][idp,0])/np.mean(datadi[k]['PV'][idp,0]-df2['avPV'][mon])<0.25:
        continue

    if np.mean(dipv[k]['env'][idp,0]/(datadi[k]['PV'][idp,0]-df2['avPV'][mon]))>=0.25:

        if np.mean(dipv[k]['env'][idp,0]/(datadi[k]['PV'][idp,0]-df2['avPV'][mon])) * np.mean(oro[k]['env'][idp,0])/np.mean(dipv[k]['env'][idp,0]) >= 0.25:
            poroid = np.append(poroid,k)
            
opvloc = dict()
ocycd = dict()
oenvd = dict()
oenvoro = dict()
oenvnoro = dict()

for h in np.arange(0,49):
    opvloc[h] = np.array([])
    ocycd[h] = np.array([])
    oenvd[h] = np.array([])
    oenvnoro[h] = np.array([])
    oenvoro[h] = np.array([])
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
    if np.any(poroid==d):
        for h in np.arange(0,49):
             opvloc[h] = np.append(opvloc[h],PV[i,h])
             ocycd[h] = np.append(ocycd[h],dipv[d][c][i,h])
             oenvd[h] = np.append(oenvd[h],dipv[d][e][i,h])
             oenvoro[h] = np.append(oenvoro[h],O[d][e][i,h])
             oenvnoro[h] = np.append(oenvnoro[h],N[d][e][i,h])
    else:             
      for h in np.arange(0,49):
        pvloc[h] = np.append(pvloc[h],PV[i,h])
        cycd[h] = np.append(cycd[h],dipv[d][c][i,h])
        envd[h] = np.append(envd[h],dipv[d][e][i,h])
        envoro[h] = np.append(envoro[h],O[d][e][i,h])
        envnoro[h] = np.append(envnoro[h],N[d][e][i,h])
        
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
fig.savefig('/atmosdyn2/ascherrmann/paper/cyc-env-PV/review/noro-high-oro-extracted-env-cyc-PV-lines-mean-t-%d-dis-%d-%s.png'%(hcyc,rdis,thresh),dpi=300,bbox_inches="tight")

plt.close('all')
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
    pvav = np.append(pvav,np.mean(opvloc[h]))
    pv10 = np.append(pv10,np.percentile(opvloc[h],10))
    pv90 = np.append(pv90,np.percentile(opvloc[h],90))
    avlic = np.append(avlic,np.mean(ocycd[h]))
    tenlic = np.append(tenlic,np.percentile(np.sort(ocycd[h]),10))
    ninetylic = np.append(ninetylic,np.percentile(np.sort(ocycd[h]),90))
    avlie = np.append(avlie,np.mean(oenvd[h]))
    tenlie = np.append(tenlie,np.percentile(np.sort(oenvd[h]),10))
    ninetylie = np.append(ninetylie,np.percentile(np.sort(oenvd[h]),90))
    cy25= np.append(cy25,np.percentile(np.sort(ocycd[h]),25))
    cy75= np.append(cy75,np.percentile(np.sort(ocycd[h]),75))
    cy50= np.append(cy50,np.percentile(np.sort(ocycd[h]),50))
    en25= np.append(en25,np.percentile(np.sort(oenvd[h]),25))
    en75= np.append(en75,np.percentile(np.sort(oenvd[h]),75))
    en50= np.append(en50,np.percentile(np.sort(oenvd[h]),50))

    avenoro = np.append(avenoro,np.mean(oenvoro[h]))
    avennoro =np.append(avennoro,np.mean(oenvnoro[h]))

    enoro10= np.append(enoro10,np.percentile(np.sort(oenvoro[h]),10))
    enoro90= np.append(enoro90,np.percentile(np.sort(oenvoro[h]),90))
    enoro25= np.append(enoro25,np.percentile(np.sort(oenvoro[h]),25))
    enoro75= np.append(enoro75,np.percentile(np.sort(oenvoro[h]),75))
    ennoro25= np.append(ennoro25,np.percentile(np.sort(oenvnoro[h]),25))
    ennoro75= np.append(ennoro75,np.percentile(np.sort(oenvnoro[h]),75))
    ennoro10= np.append(ennoro10,np.percentile(np.sort(oenvnoro[h]),10))
    ennoro90= np.append(ennoro90,np.percentile(np.sort(oenvnoro[h]),90))


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
fig.savefig('/atmosdyn2/ascherrmann/paper/cyc-env-PV/review/noro-high-oro-only-env-cyc-PV-lines-mean-t-%d-dis-%d-%s.png'%(hcyc,rdis,thresh),dpi=300,bbox_inches="tight")

plt.close('all')
