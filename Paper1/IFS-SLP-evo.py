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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle
import xarray as xr

import cartopy
import matplotlib.gridspec as gridspec
import functools

def colbar(cmap,minval,maxval,nlevels):
    maplist = [cmap(i) for i in range(cmap.N)]
    newmap = ListedColormap(maplist)
    norm = BoundaryNorm(pvr_levels,cmap.N)
    return newmap, norm

CT = 'MED'

pload = '/atmosdyn2/ascherrmann/010-IFS/ctraj/' + CT + '/use/'

f = open(pload + 'PV-data-'+CT+'dPSP-100-ZB-800PVedge-0.3-400-correct-distance.txt','rb')
data = pickle.load(f)
f.close()

f = open('/atmosdyn2/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','rb')
ldata = pickle.load(f)
f.close()

NORO = xr.open_dataset('/atmosdyn2/ascherrmann/010-IFS/data/IFSORO')
ZB = NORO['ZB'].values[0,0]
oro = data['oro']
datadi = data['rawdata']
dipv = data['dipv']

rdis = 400

ps = '/atmosdyn2/ascherrmann/paper/cyc-env-PV/'
LON = np.arange(-180,180.1,0.4)
LAT = np.arange(0,90.1,0.4)

text = ['a)','b)']
for mon in ldata.keys():
    if mon!='DEC17' and mon!='JUN18':
        continue

    for ID in ldata[mon].keys():
        if (mon=='DEC17' and ID==73) or (mon=='JUN18' and ID==111):
            
            fig,ax=plt.subplots()
            slp = ldata[mon][ID]['SLP']
            hslp = ldata[mon][ID]['hSLP']
             
            ax.set_xticks(np.arange(-96,97,6))
            ax.set_xticklabels(np.arange(-96,97,6))

            ax.set_xlim(np.min(hslp),np.max(hslp))
            ax.set_xlabel('time to SLP min [h]')
            ax.set_ylabel('SLP [hPa]')

            ax.plot(hslp,slp,color='k')
            figname = ps + 'SLP-evo' + mon + '-' + str(int(ID))  + '.png'
            fig.savefig(figname,dpi=300,bbox_inches="tight")
            plt.close()


