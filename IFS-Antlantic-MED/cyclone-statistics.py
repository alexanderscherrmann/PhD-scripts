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

rdis = 400
pload = '/home/ascherrmann/010-IFS/traj/MED/use/'
CT = 'MED'
f = open(pload + 'PV-data-MEDdPSP-100-ZB-800PVedge-0.3-%d.txt'%rdis,'rb')
PVdata = pickle.load(f)
f.close()

f = open('/home/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','rb')
locdata =  pickle.load(f)
f.close()

t = np.arange(-168,121)
SLPA = np.zeros((56,len(t)))
countert = np.zeros(len(t))
prematurelifespan = np.array([])
postmaturelifespan = np.array([])
me = np.array([])
fig,axes = plt.subplots()
axes = [axes,0]
ax = axes[0]
for u,k in enumerate(PVdata['rawdata'].keys()):
    mon = PVdata['mons'][u]
    q = int(k[-3:])
#    h = np.where(t==locdata[mon][q]['hzeta'][0])[0][0]
#    i = len(locdata[mon][q]['hzeta'])
#    prematurelifespan = np.append(prematurelifespan,abs(locdata[mon][q]['hzeta'][0]))
#    postmaturelifespan = np.append(postmaturelifespan,locdata[mon][q]['hzeta'][-1])
    h = np.where(t==locdata[mon][q]['hSLP'][0])[0][0]
    
    i = len(locdata[mon][q]['hSLP'])
    prematurelifespan = np.append(prematurelifespan,abs(locdata[mon][q]['hSLP'][0]))
    postmaturelifespan = np.append(postmaturelifespan,locdata[mon][q]['hSLP'][-1])
    
    if h+i>len(t):
        countert[h:]+=1
        SLPA[u,h:]=locdata[mon][q]['SLP'][:len(t[h:])]
    else:
        countert[h:h+i]+=1
        SLPA[u,h:h+i]=locdata[mon][q]['SLP']

#    ax.plot(locdata[mon][q]['hSLP'],locdata[mon][q]['SLP'],color='k',linewidth=0.2,zorder=100)
    plid = np.where(SLPA[u]!=0)[0]
    ax.plot(t[plid],SLPA[u,plid],color='k',linewidth=0.2,zorder=100)
    me = np.append(me,locdata[mon][q]['SLP'][abs(locdata[mon][q]['hSLP'][0]).astype(int)])
slpav = np.array([])
slp10 = np.array([])
slp90 = np.array([])
#fig.savefig(pload[:-13] + 'allSLP.png',dpi=300,bbox_inches='tight') 
#plt.close('all')

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


#fig,axes = plt.subplots()#2,1,sharex=True)
#axes = [axes,0]
#axes = axes.flatten()
axes[0].plot(t,slp10,color='grey')
axes[0].plot(t,slp90,color='grey')
axes[0].plot(t,slpav,color='k')
axes[0].fill_between(t,slp10,slp90,alpha=0.5,color='grey')
axes[0].set_ylabel('SLP [hPa]')

axes[0].axvline(-1 * np.mean(prematurelifespan),color='k')
axes[0].axvline(np.mean(postmaturelifespan),color='k')
axes[0].set_ylim(970,1010)

axes[0].text(0.03, 0.95, 'c)', transform=axes[0].transAxes,fontsize=12, fontweight='bold',va='top')
#axes[1].text(0.03, 0.95, 'd)', transform=axes[1].transAxes,fontsize=12, fontweight='bold',va='top')
#
axes[0].set_xlim(-48,48)
#axes[1].plot(t,countert,color='k')
#
#axes[1].axvline(-1 * np.mean(prematurelifespan),color='k')
#axes[1].axvline(np.mean(postmaturelifespan),color='k')
#
axes[0].set_xlabel('time to SLP min [h]')
axes[0].set_xticks(ticks=np.arange(-48,48,6))
#axes[1].set_xlim(-48,48)
##axes[1].set_xticks(ticks=np.arange(-120,121,12))
##axes[1].set_xticklabels(labels=np.arange(-120,121,12))
#axes[1].set_ylabel('number of cyclones')
#axes[1].set_ylim(1,60)
plt.subplots_adjust(left=0.1,hspace=0,wspace=0)
fig.savefig('/home/ascherrmann/010-IFS/cyclone-statistics-IFS.png',dpi=300,bbox_inches="tight")
plt.close('all')

