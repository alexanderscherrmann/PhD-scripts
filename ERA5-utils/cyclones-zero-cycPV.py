import numpy as np
import xarray as xr
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.stats.stats import pearsonr
import pickle
sys.path.append('/home/raphaelp/phd/scripts/basics/')

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
psave = '/home/ascherrmann/009-ERA-5/MED/orocyclones/'

f = open(pload + 'PV-data-dPSP-100-ZB-800.txt','rb')
PVdata = pickle.load(f)
f.close()

datadi = PVdata['rawdata']
dipv = PVdata['dipv']
ORO = PVdata['oro']

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
minSLP = np.array([])
for k in range(len(ID)):
    avaID=np.append(avaID,ID[k][0].astype(int))
    minSLP = np.append(minSLP,SLP[k][abs(hourstoSLPmin[k][0]).astype(int)])


pvival = np.array([-100,0.2,0.5,0.75,100])

anomaly = dict()
mature = dict()
layercounter = dict()
cycc =0

pinvals = np.arange(700,925.1,12.5)
plvlcounter = dict()

zerocyc = np.array([])
split = ['cyc','env']
linestyle = ['-',':']
fsl=6
oroids = np.array([])
for ll,k in enumerate(datadi.keys()):
    q = np.where(avaID==int(k))[0][0]
    d = k

#    if (hourstoSLPmin[q][0]>-6):
#        continue

    PV = datadi[d]['PV'][:,0]
    P = datadi[d]['P'][:,0]
    PV[np.where(PV<-4)]=0
    i = np.where((PV>=0.75))[0]# & (P<925))[0]

    if np.all(np.mean(dipv[d]['cyc'][i,:],axis=0)==0):
        zerocyc = np.append(zerocyc,k)

CT ='MED'
if CT=='MED':
    minpltlonc = -10
    maxpltlonc = 45
    minpltlatc = 25
    maxpltlatc = 50
    steps = 5

lonticks=np.arange(minpltlonc, maxpltlonc+1,steps)
latticks=np.arange(minpltlatc, maxpltlatc+1,steps)

for q,k in enumerate(zerocyc):
    if q%12==0:
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
        ax.coastlines()
    qq = np.where(avaID==int(k))[0][0]
    ax.plot(lon[qq],lat[qq])
    ax.scatter(lon[qq][abs(hourstoSLPmin[qq][0]).astype(int)],lat[qq][abs(hourstoSLPmin[qq][0]).astype(int)],marker='.',s=6)

    if q%12==11:
        ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
        ax.set_yticks(latticks, crs=ccrs.PlateCarree());
        ax.set_xticklabels(labels=lonticks,fontsize=10)
        ax.set_yticklabels(labels=latticks,fontsize=10)
        
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        
        ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
        
        fig.savefig('/home/ascherrmann/009-ERA-5/MED/' + 'zero-cyclonic-cyclones-%03d'%q + '.png',dpi=300,bbox_inches="tight")
        plt.close('all')
print(len(zerocyc))



