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

parser = argparse.ArgumentParser(description="plot accumulated average PV gain that is associated with the cyclone and the environment")
parser.add_argument('rdis',default='',type=int,help='distance from center in km that should be considered as cyclonic')
args = parser.parse_args()
rdis = int(args.rdis)

pload = '/home/ascherrmann/009-ERA-5/MED/traj/use/'
f = open(pload + 'PV-data-dPSP-100-ZB-800-2-%d.txt'%rdis,'rb')
PVdata = pickle.load(f)
f.close()

dipv = PVdata['dipv']
datadi = PVdata['rawdata']
dit = PVdata['dit']

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
maturedates = np.array([])
for k in range(len(ID)):
    avaID=np.append(avaID,ID[k][0].astype(int))
    maturedates = np.append(maturedates,dates[k][abs(hourstoSLPmin[k][0]).astype(int)])

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

counters = dict()
counters[60] = dict()
counters[85] = dict()
for q in range(6):
    counters[60][q] = np.zeros((361,721))
    counters[85][q] = np.zeros((361,721))

size = 1
for h in np.arange(0,49):
    pvloc[h] = np.array([])
    cycd[h] = np.array([])
    envd[h] = np.array([])

identifi = np.array([])
fib = 0.6
tb = 0.75
thb =0.85
fb=0.85
tb=0.85
hcyc = 12

advid0 = np.array([])
advid12 = np.array([])
envid0= np.array([])
envid12= np.array([])

LON = np.arange(-180,180.1,0.5)
LAT = np.arange(-90,90.1,0.5)

for ll,k in enumerate(dipv.keys()):
    q = np.where(avaID==int(k))[0][0]
    CLON = lon[q][abs(hourstoSLPmin[q][0]).astype(int)]
    CLAT = lat[q][abs(hourstoSLPmin[q][0]).astype(int)]

    if(CLON%0.5!=0):
        CLON = np.round(CLON,0)
    if(CLAT%0.5!=0):
        CLAT= np.round(CLAT,0)

    d = k
    pre = PVdata['rawdata'][d]['P']
    PV = PVdata['rawdata'][d]['PV']
    i = np.where(PV[:,0]>=0.75)[0]

    pvend = PV[i,0]
    pvstart = PV[i,-1]
    cypv = dipv[d][c][i,0]
    enpv = dipv[d][e][i,0]
    cy = np.mean(cypv)

    uu = np.where(LON==CLON)[0][0]
    ll = np.where(LAT==CLAT)[0][0]

#    if np.mean(cypv/pvend)>tb:
#        counters[85][0][ll,uu]+=1
    if np.mean(enpv/pvend)>tb:
        counters[85][1][ll,uu]+=1
        envid0 = np.append(envid0,q)
    if np.mean(pvstart/pvend)>tb:
        counters[85][2][ll,uu]+=1
        advid0 = np.append(advid0,q)

###################################
    if (hourstoSLPmin[q][0]>(-1*hcyc)):
        continue

    if len(i)<100:
        continue
##################################
#    if np.mean(cypv/pvend)>tb:
#        counters[85][3][ll,uu]+=1
    if np.mean(enpv/pvend)>tb:
        counters[85][4][ll,uu]+=1
        envid12 = np.append(envid12,q)

    if np.mean(pvstart/pvend)>tb:
        counters[85][5][ll,uu]+=1
        advid12 = np.append(advid12,q)

minpltlonc = -10
maxpltlonc = 45
minpltlatc = 25
maxpltlatc = 50
steps = 5

lonticks=np.arange(minpltlonc, maxpltlonc+1,steps*3)
latticks=np.arange(minpltlatc, maxpltlatc+1,steps*2)

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(nrows=2, ncols=2)
axes = []
for k in range(2):
  for l in range(2):
    axes.append(fig.add_subplot(gs[k,l],projection=ccrs.PlateCarree()))

ids = [envid0,advid0,envid12,advid12]
for k,ax in enumerate(axes):
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)
    for u in ids[k]:
        u = u.astype(int)
#        ax.plot(lon[u],lat[u],color='k',linewidth=0.2)
        ax.scatter(lon[u][0],lat[u][0],color='b',marker='x',s=10)
        ax.scatter(lon[u][abs(hourstoSLPmin[u][0]).astype(int)],lat[u][abs(hourstoSLPmin[u][0]).astype(int)],marker='o',color='r',s=10)
        longrids=np.arange(-180,180,5)
        latgrids=np.arange(-90,90,5)

    ax.gridlines(xlocs=longrids, ylocs=latgrids, linestyle='--',color='grey',zorder=1)
    ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
    ax.set_yticks(latticks, crs=ccrs.PlateCarree());
    ax.set_xticklabels(labels=lonticks,fontsize=10)
    ax.set_yticklabels(labels=latticks,fontsize=10)

    if k==1 or k==3:
        ax.set_yticklabels([])

#    ax.xaxis.set_major_formatter(LongitudeFormatter())
#    ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
plt.subplots_adjust(top=0.6,bottom=0.1,hspace=0.15,wspace=0)
fig.savefig('/home/ascherrmann/009-ERA-5/MED/env-adv-tracks.png',dpi=300,bbox_inches="tight")

plt.close('all')
