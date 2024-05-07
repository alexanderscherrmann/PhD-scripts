import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy
import matplotlib.gridspec as gridspec
import functools
import cartopy.crs as ccrs

import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')

from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import xarray as xr


p = '/home/ascherrmann/009-ERA-5/MED/'
NORO = xr.open_dataset(p + 'data/NORO')
LON = NORO['lon'].values
LAT = NORO['lat'].values

lons = np.where((LON>=-5) & (LON<=42))[0]
lats = np.where((LAT>=30) & (LAT<=48))[0]

LSM = NORO['OL'][0,lats,lons]

a = 'wo'
f = open(p + 'climatologyPV-' + a + '-cyclones.txt','rb')
d = pickle.load(f)
f.close()

a = 'w'
f = open(p + 'climatologyPV-' + a + '-cyclones.txt','rb')
d2 = pickle.load(f)
f.close()

lon = np.arange(-5,42.1,0.5)
lat = np.arange(30,48.1,0.5)

con = dict()
for m in range(0,13):
    con[m] = np.zeros((len(lat),len(lon)))
    
locPV = d['locPV']
loccount = d['loccount']

locPV2 = d2['locPV']
loccount2 = d2['loccount']


ovseaPVw = dict()
ovseaPVwo = dict()
countwo = 0
countw =  0
ovseac = len(np.where(LSM==0)[0])

for m in np.arange(0,13):
    ovseaPVw[m] = 0
    ovseaPVwo[m] = 0
    if m==0:
        continue
    for k in locPV[m].keys():
        if loccount[m][k]==0:
            continue
        lo = np.where(lon==float(k[:-5]))[0][0]
        la = np.where(lat==float(k[-4:]))[0][0]
        con[m][la,lo] = locPV[m][k]/loccount[m][k]-locPV2[m][k]/loccount2[m][k]

        if LSM[la,lo] == 0:
            ovseaPVw[m] += locPV2[m][k]/loccount2[m][k]
            ovseaPVwo[m] += locPV[m][k]/loccount[m][k]
            ovseaPVw[0] += locPV2[m][k]/loccount[m][k]
            ovseaPVwo[0] += locPV2[m][k]/loccount2[m][k]
            countwo += loccount[m][k]
            countw += loccount2[m][k]

    ovseaPVw[m]/=ovseac
    ovseaPVwo[m]/=ovseac

ovseaPVw[0] /= ovseac * 12
ovseaPVwo[0] /= ovseac * 21

print(np.round(np.array(list(ovseaPVw.values())),3))
print(np.round(np.array(list(ovseaPVwo.values())),3))
print(np.mean(np.round(np.array(list(ovseaPVwo.values())[1:]),3)))
print(np.mean(np.round(np.array(list(ovseaPVw.values())[1:]),3)))

np.savetxt(p +'clim-avPV.txt',np.round(np.array(list(ovseaPVwo.values())[1:]),3),fmt='%.3f',delimiter=' ',newline='\n')

level = np.arange(-0.15,0.16,0.03)
cmap = matplotlib.cm.BrBG
cmap.set_over('navy')
cmap.set_under('palegoldenrod')

labs = ['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

for m in range(1,13):
    fig = plt.figure(figsize=(6,4))
    gs = gridspec.GridSpec(ncols=1, nrows=1)
    ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')

    cf = ax.contourf(lon,lat,con[m],cmap=cmap,levels=level,extend='both')
    ax.set_extent([-5,42,30,48],ccrs.PlateCarree())
    
    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(cf, ticks=level,cax=cbax)
    func=resize_colorbar_vert(cbax, ax, pad=0.01, size=0.02)
    
    fig.canvas.mpl_connect('draw_event', func)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel('PVU',fontsize=8)

    
    fig.savefig(p + labs[m] + '-avPVmap-difference.png',dpi=300,bbox_inches='tight')
    plt.close('all')

