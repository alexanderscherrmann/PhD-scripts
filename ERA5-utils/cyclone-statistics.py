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
import pandas as pd

rdis = 400
pload = '/home/ascherrmann/009-ERA-5/MED/ctraj/use/'
f = open(pload + 'PV-data-dPSP-100-ZB-800-2-%d-correct-distance.txt'%rdis,'rb')
PVdata = pickle.load(f)
f.close()
savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']
var = []

for u,x in enumerate(savings):
    f = open(pload[:-10] + pload[-9:-4]  + x + 'furthersel.txt',"rb")
    var.append(pickle.load(f))
    f.close()
    
SLP = var[0]
lon = var[1]
lat = var[2]
ID = var[3]
hourstoSLPmin = var[4]
dates = var[5]
avaID = np.array([])
maturedates = np.array([])

for k in range(len(ID)):
    avaID=np.append(avaID,ID[k][0].astype(int))
    maturedates = np.append(maturedates,dates[k][abs(hourstoSLPmin[k][0]).astype(int)])

df = pd.read_csv('/home/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')
df = df.loc[df['ntraj075']>=200]
ID = df['ID'].values

t = np.arange(-120,121)
SLPA = np.zeros((len(ID),len(t)))
countert = np.zeros(len(t))
prematurelifespan = np.array([])
postmaturelifespan = np.array([])
u = 0
for rr,k in enumerate(PVdata['rawdata'].keys()):
    if np.all(ID!=int(k)):
        continue
    q = np.where(avaID==int(k))[0][0]
    h = np.where(t==hourstoSLPmin[q][0])[0][0]
    i = len(hourstoSLPmin[q])
    prematurelifespan = np.append(prematurelifespan,abs(hourstoSLPmin[q][0]))
    postmaturelifespan = np.append(postmaturelifespan,hourstoSLPmin[q][-1])
    if h+i>len(t):
        countert[h:]+=1
        SLPA[u,h:]=SLP[q][:len(t[h:])]
    else:
        countert[h:h+i]+=1
        SLPA[u,h:h+i]=SLP[q]
    u+=1 
slpav = np.array([])
slp10 = np.array([])
slp90 = np.array([])

SLPAA = np.delete(SLPA,np.where(countert==0)[0],axis=1)
t = np.delete(t,np.where(countert==0)[0])
countert = np.delete(countert,np.where(countert==0)[0])

for u,r in enumerate(t):
    if np.all(SLPAA[:,u]==0):
        continue
    else:
        i = np.where(SLPAA[:,u]!=0)[0] 

    slpav = np.append(slpav,np.mean(SLPAA[i,u]))
    slp10 = np.append(slp10,np.percentile(SLPAA[i,u],10))
    slp90 = np.append(slp90,np.percentile(SLPAA[i,u],90))
    
fig,axes = plt.subplots()#2,1,sharex=True)
axes = [axes,0]
#axes = axes.flatten()
#axes.append(0)
axes[0].plot(t,slp10,color='grey')
axes[0].plot(t,slp90,color='grey')
axes[0].plot(t,slpav,color='k')
axes[0].fill_between(t,slp10,slp90,alpha=0.5,color='grey')
axes[0].set_ylabel('SLP [hPa]')
axes[0].set_ylim(990,1020)

axes[0].axvline(-1 * np.mean(prematurelifespan),color='k')
axes[0].axvline(np.mean(postmaturelifespan),color='k')
axes[0].text(0.03, 0.95, 'a)', transform=axes[0].transAxes,fontsize=12, fontweight='bold',va='top')
axes[0].set_xlim(-48,48)
#axes[1].plot(t,countert,color='k')
#
#axes[1].set_yscale('log')
axes[0].set_xlabel('time to SLP min [h]')
axes[0].set_xticks(ticks=np.arange(-48,49,12))
axes[0].set_xticklabels(labels=np.arange(-48,49,12))
#axes[1].set_xlim(-96,96)
#axes[1].set_ylabel('number of cyclones')
#axes[1].set_ylim(10,5000)
#axes[1].axvline(-1 * np.mean(prematurelifespan),color='k')
#axes[1].axvline(np.mean(postmaturelifespan),color='k')
#axes[1].text(0.03, 0.95, 'b)', transform=axes[1].transAxes,fontsize=12, fontweight='bold',va='top')
#plt.subplots_adjust(left=0.1,hspace=0,wspace=0)
fig.savefig('/home/ascherrmann/009-ERA-5/MED/cyclone-statistics-ERA5.png',dpi=300,bbox_inches="tight")
plt.close('all')

