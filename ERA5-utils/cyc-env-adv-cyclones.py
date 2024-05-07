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

pload = '/home/ascherrmann/009-ERA-5/MED/traj/use/'
f = open(pload + 'PV-data-dPSP-100-ZB-800.txt','rb')
PVdata = pickle.load(f)
f.close()
dipv = PVdata['dipv']
datadi = PVdata['rawdata']

both = np.array([])
adv = np.array([])
cyclonic = np.array([])
environmental = np.array([])

savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']
var = []

for u,x in enumerate(savings):
    f = open(pload[:-4] + x + 'furthersel.txt',"rb")
    var.append(pickle.load(f))
    f.close()
SLP = var[0]
lon = var[1]
lat = var[2]
ID = var[3]
hourstoSLPmin = var[4]
dates = var[5]
avaID = np.array([])
for k in range(len(ID)):
    avaID=np.append(avaID,ID[k][0].astype(int))

for k in dipv.keys():
    q = np.where(avaID==int(k))[0][0]
#    if (hourstoSLPmin[q][0]>-6):
#        continue

    idp = np.where(PVdata['rawdata'][k]['PV'][:,0]>=0.75)[0]
    PVend = PVdata['rawdata'][k]['PV'][idp,0]
    cyc = dipv[k]['cyc'][idp,0]
    env = dipv[k]['env'][idp,0]
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
for k in cyclonic:
    tralon = np.mean(datadi[k]['lon'][:,0])
    tralat = np.mean(datadi[k]['lat'][:,0])
    cycol = np.append(cycol,np.mean(datadi[k]['OL'][:,0]))

    ax.scatter(tralon,tralat,color='r',marker='.',zorder=100,s=0.4)

for k in adv:
    tralon = np.mean(datadi[k]['lon'][:,0])
    tralat = np.mean(datadi[k]['lat'][:,0])
    ax.scatter(tralon,tralat,color='b',marker='.',zorder=10,s=0.4)
    advol = np.append(advol,np.mean(datadi[k]['OL'][:,0]))

for k in environmental:
    tralon = np.mean(datadi[k]['lon'][:,0])
    tralat = np.mean(datadi[k]['lat'][:,0])
    ax.scatter(tralon,tralat,color='k',marker='.',zorder=50,s=0.4)
    envol = np.append(envol,np.mean(datadi[k]['OL'][:,0]))
        
for k in both:
    tralon = np.mean(datadi[k]['lon'][:,0])
    tralat = np.mean(datadi[k]['lat'][:,0])
    ax.scatter(tralon,tralat,color='slategrey',marker='.',zorder=15,s=0.4)
    bothol = np.append(bothol,np.mean(datadi[k]['OL'][:,0]))

print(len(np.where(cycol<0.5)[0])/len(cycol))
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

psave = '/home/ascherrmann/009-ERA-5/MED/'
fig.savefig(psave + 'all-loc-of-cyclone-types.png',dpi=300,bbox_inches="tight")
plt.close('all')
