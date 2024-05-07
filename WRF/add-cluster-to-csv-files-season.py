import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from wrf import interplevel as intp
from netCDF4 import Dataset as ds
import os
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec

import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert

import pickle

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
boxnames = ['balearic','adriaticN','adriaticS','ionian-aegean','sicily','cyprus','black','africaE','africaC','tyrric','genua','centralMed','greece','belowgreece']

boxes = [[-4,9,37.5,42], #bal
        [12,19.5,42,46], #adN
        [15.5,21,38,42.5], #adS
        [20.5,25,31.5,37], #ioae
        [10,16.5,34.5,38], #sic
        [30,36.5,30,38], #cyp
        [28,42,40,50], #bla
        [-4,10,30,38], #afE
        [10,21,30,35], #afC
        [8.5,16.5,37.5,42], #tyr
        [0,12,41.5,46], #gen
        [16,21,34.5,38.5], #cen
        [20.5,28.5,36.5,42], #gre
        [24.5,30.5,30,37]] #bel

### only for plotting!!!!
#boxes = [[-4,8.5,38,42], #bal
#        [12,19.5,42,46], #adN
#        [16,21,38,42], #adS
#        [21,24.5,31.5,37], #ioae
#        [10,16,35,38], #sic
#        [30,36.5,30,38], #cyp
#        [28,42,40,50], #bla
#        [-4,10,30,38], #afE
#        [10,21,30,35], #afC
#        [8.5,16,38,42], #tyr
#        [0,12,42,46], #gen
#        [16,21,35,38], #cen
#        [21,28,37,42], #gre
#        [24.5,30,30,37]] #bel


fig = plt.figure(figsize=(6,4))
gs = gridspec.GridSpec(ncols=1, nrows=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0,edgecolor='black',facecolor='lightgrey',alpha=0.7)
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
for n,b in zip(boxnames,boxes):
    lab,lat,lol,lor = b[2],b[3],b[0],b[1]
    ax.plot([lol,lor],[lab,lab],color='k')
    ax.plot([lol,lor],[lat,lat],color='k')
    ax.plot([lol,lol],[lab,lat],color='k')
    ax.plot([lor,lor],[lab,lat],color='k')
#    ax.text((lor+lol)/2-2,(lat+lab)/2-1,n,fontsize=6,color='b')


df = pd.read_csv(ps + 'SON-intense-cyclones.csv')
lon = df['lon'].values; lat = df['lat'].values
region = np.array([])
for q,lo,la in zip(range(len(lon)),lon,lat):
    for n,b in zip(boxnames,boxes):
        lab,lat,lol,lor = b[2],b[3],b[0],b[1]
        if la<lat and la>lab and lo<lor and lo>lol:
            region = np.append(region,n)
            break
    if len(region)==q:
        region = np.append(region,'none')
        print(lo,la)
        #ax.scatter(lo,la,color='r',s=20)

df['region'] = region
fig.savefig('/home/ascherrmann/boxes.png',bbox_inches='tight',dpi=300)
df.to_csv(ps + 'SON-intense-cyclones.csv',index=False)


