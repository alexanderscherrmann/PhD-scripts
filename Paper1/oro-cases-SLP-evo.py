import numpy as np
import os
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
import matplotlib.gridspec as gridspec
import cartopy
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle
import xarray as xr
import pandas as pd


def colbar(cmap,minval,maxval,nlevels,levels):
    maplist = [cmap(i) for i in range(cmap.N)]
    newmap = ListedColormap(maplist)
    norm = BoundaryNorm(levels,cmap.N)
    return newmap, norm

rdis = 400
pload = '/atmosdyn2/ascherrmann/009-ERA-5/MED/ctraj/use/'
ps = '/atmosdyn2/ascherrmann/paper/cyc-env-PV/'

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


datadi = PVdata['rawdata']
avaID = np.array([])
maturedates = np.array([])

for k in range(len(ID)):
    avaID=np.append(avaID,ID[k][0].astype(int))
    maturedates = np.append(maturedates,dates[k][abs(hourstoSLPmin[k][0]).astype(int)])

df = pd.read_csv('/atmosdyn2/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')
df = df.loc[df['ntraj075']>=200]
hcyc = 12
df = df.loc[df['htminSLP']>=hcyc]

for rr,k in enumerate(PVdata['rawdata'].keys()):
    if k!='108215' and k!='119896':
        continue

    fig,ax = plt.subplots()
    I = np.where(avaID==int(k))[0][0]
    slp = SLP[I]
    t = np.arange(slp.size)-np.arange(slp.size)[np.argmin(slp)]
    ax.plot(t,slp,color='k')
    ax.set_xticks(np.arange(-48,49,6))
    ax.set_xticklabels(np.arange(-48,49,6))
    ax.set_xlabel('time to SLP min [h]')
    ax.set_ylabel('SLP [hPa]')
    ax.set_xlim(np.min(t),np.max(t))
    figname = ps + 'SLP-evo' + str(k)  + '.png'
    fig.savefig(figname,dpi=300,bbox_inches="tight")
    plt.close('all')


