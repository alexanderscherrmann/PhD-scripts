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
import os


CT = 'MED'
pload = '/home/ascherrmann/010-IFS/traj/' + CT + '/use/'
plload = '/home/ascherrmann/010-IFS/traj/' + CT + '/'

MON = np.array([])
for d in os.listdir(plload):
    if(d.startswith('traend-')):
        MON = np.append(MON,d)

traj = np.array([])
for d in os.listdir(pload):
    if(d.startswith('trajectories-mature-')):
            traj = np.append(traj,d)

traj = np.sort(traj)
MON = np.sort(MON)


f = open(pload + 'PV-data-' + CT + 'dPSP-100-ZB-800.txt','rb')
PVdata = pickle.load(f)
f.close()

dipv = PVdata['dipv']
datadi = PVdata['rawdata']

both = np.array([])
adv = np.array([])
cyclonic = np.array([])
environmental = np.array([])

f = open('/home/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','rb')
td = pickle.load(f)
f.close()
for uyt, txt in enumerate(traj):

    montmp = MON[uyt][-9:-4]
    idtmp = int(txt[-10:-4])

    date=txt[-25:-14]
    date=date+'-%03d'%idtmp
    htzeta = td[montmp][idtmp]['hzeta']

    if (htzeta[0]>-6):
        continue

    k = date

    idp = np.where(PVdata['rawdata'][k]['PV'][:,0]>=0.75)[0]
    PVend = PVdata['rawdata'][k]['PV'][idp,0]
    cyc = dipv[k]['cyc']['APVTOT'][idp,0]
    env = dipv[k]['env']['APVTOT'][idp,0]
    tot = cyc + env
    cycm = np.mean(cyc)
    envm = np.mean(env)
    if np.mean(tot/PVend)>0.5:
      if((envm>0) & (cycm>0)):
        if ((cycm/envm)>1.3):
            cyclonic = np.append(cyclonic,k)
        elif (envm/cycm>1.3):
            environmental = np.append(environmental,k)
        else:
            both = np.append(both,k)
          
      elif ((envm<0) & (cycm<0)):
          adv = np.append(adv,k)
          
      elif((cycm<0) & (envm>0)):
          environmental = np.append(environmental,k)
          
      elif((cycm>0) & (envm<0)):
          cyclonic = np.append(cyclonic,k)
          
      else:
          adv = np.append(adv,k)
    else:
        adv = np.append(adv,k)

cycol = np.array([])
envol = np.array([])
bothol = np.array([])
advol = np.array([])

fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.coastlines()
size = 5.
for k in cyclonic:
    tralon = np.mean(datadi[k]['lon'][:,0])
    tralat = np.mean(datadi[k]['lat'][:,0])
    cycol = np.append(cycol,np.mean(datadi[k]['OL'][:,0]))

    ax.scatter(tralon,tralat,color='r',marker='.',zorder=100,s=size)

for k in adv:
    tralon = np.mean(datadi[k]['lon'][:,0])
    tralat = np.mean(datadi[k]['lat'][:,0])
    ax.scatter(tralon,tralat,color='b',marker='.',zorder=10,s=size)
    advol = np.append(advol,np.mean(datadi[k]['OL'][:,0]))

for k in environmental:
    tralon = np.mean(datadi[k]['lon'][:,0])
    tralat = np.mean(datadi[k]['lat'][:,0])
    ax.scatter(tralon,tralat,color='k',marker='.',zorder=50,s=size)
    envol = np.append(envol,np.mean(datadi[k]['OL'][:,0]))
        
for k in both:
    tralon = np.mean(datadi[k]['lon'][:,0])
    tralat = np.mean(datadi[k]['lat'][:,0])
    ax.scatter(tralon,tralat,color='slategrey',marker='.',zorder=15,s=size)
    bothol = np.append(bothol,np.mean(datadi[k]['OL'][:,0]))

#print(len(np.where(cycol<0.5)[0])/len(cycol))
#print(len(np.where(envol<0.5)[0])/len(envol))
#print(len(np.where(advol<0.5)[0])/len(advol))
#print(len(np.where(bothol<0.5)[0])/len(bothol))

t = len(cycol) + len(envol) + len(advol) + len(bothol)
print(len(cycol)/t)
print(len(envol)/t)
print(len(advol)/t)
print(len(bothol)/t)


minpltlonc = -10
maxpltlonc = 45
minpltlatc = 25
maxpltlatc = 50
steps = 5


lonticks=np.arange(minpltlonc, maxpltlonc,steps)
latticks=np.arange(minpltlatc, maxpltlatc,steps)

ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
ax.set_yticks(latticks, crs=ccrs.PlateCarree());
ax.set_xticklabels(labels=lonticks,fontsize=10)
ax.set_yticklabels(labels=latticks,fontsize=10)

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())

psave = '/home/ascherrmann/010-IFS/'
fig.savefig(psave + 'loc-of-cyclone-types-IFS.png',dpi=300,bbox_inches="tight")
plt.close('all')
