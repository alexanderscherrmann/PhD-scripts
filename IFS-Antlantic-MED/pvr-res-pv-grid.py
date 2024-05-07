import numpy as np
import os
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')

import xarray as xr
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

import cartopy
import matplotlib.gridspec as gridspec



def colbar(cmap,minval,maxval,nlevels):
    maplist = [cmap(i) for i in range(cmap.N)]
    newmap = ListedColormap(maplist)
    norm = BoundaryNorm(pvr_levels,cmap.N)
    return newmap, norm

LON = np.round(np.linspace(-180,180,901),1)
LAT = np.round(np.linspace(0,90,226),1)

def find_nearest_grid_point(lon,lat):

    dlon = LON-lon
    dlat = LAT-lat

    lonid = np.where(abs(dlon)==np.min(abs(dlon)))[0][0]
    latid = np.where(abs(dlat)==np.min(abs(dlat)))[0][0]

    return lonid,latid


CT = 'MED'
pload = '/home/ascherrmann/010-IFS/ctraj/MED/use/'
psave = '/home/ascherrmann/010-IFS/'

f = open(pload+'PV-data-' + CT + 'dPSP-100-ZB-800PVedge-0.3-400-correct-distance.txt','rb')
data = pickle.load(f)
f.close()

datadi = data['rawdata']
gridmap = dict()
labs = helper.traced_vars_IFS()
pvrs = np.append(np.append(np.append(np.append(np.append(labs[8:],'PVRTOT'),'DELTAPV'),'RES'),'locres'),'PV')

for pv in np.append(pvrs,'locres2'):
    gridmap[pv] = np.zeros((LAT.size,LON.size))

counter = np.zeros((LAT.size,LON.size))
PVedge=0.75




prodi = dict()
for q, date in enumerate(datadi.keys()):
    prodi[date] = dict()
    idp = np.where(datadi[date]['PV'][:,0]>=PVedge)[0]
    tralon = datadi[date]['lon'][idp]
    tralat = datadi[date]['lat'][idp]
    datadi[date]['DELTAPV'] = np.zeros(datadi[date]['PV'].shape)
    datadi[date]['DELTAPV'][:,1:] = datadi[date]['PV'][:,:-1]-datadi[date]['PV'][:,1:]
    datadi[date]['RES'] = np.zeros(datadi[date]['PV'].shape)

    datadi[date]['PVRTOT'] = np.zeros(datadi[date]['PV'].shape)
    for pv in labs[8:]:
        datadi[date]['PVRTOT']+=datadi[date][pv]

    datadi[date]['RES'][:,1:] = np.cumsum(datadi[date]['PVRTOT'][:,1:],axis=1)-np.cumsum(datadi[date]['DELTAPV'][:,1:],axis=1)
    datadi[date]['locres'] = np.zeros(datadi[date]['PV'].shape)
    datadi[date]['locres'][:,1:] = datadi[date]['PVRTOT'][:,1:] - datadi[date]['DELTAPV'][:,1:]

    datadi[date]['locres2'] = np.zeros(datadi[date]['PV'].shape)
    datadi[date]['locres2'][:,1:] = abs(datadi[date]['PVRTOT'][:,1:] - datadi[date]['DELTAPV'][:,1:])

    for k in range(len(idp)):
        for l in range(len(tralon[0])):
            lon = tralon[k,l]
            lat = tralat[k,l]
            lonid,latid = find_nearest_grid_point(lon,lat)
            for pv in np.append(pvrs[:-1],'locres2'):
                gridmap[pv][latid,lonid]+=datadi[date][pv][idp[k],l]
            latlonid = '%03d-%03d'%(latid,lonid)
            if (latlonid not in prodi[date].keys()):
                prodi[date][latlonid] = np.array([])
            gridmap['PV'][latid,lonid]+=datadi[date]['PV'][idp[k],-1]

            counter[latid,lonid]+=1

            tmp = np.array([])
            for pv in labs[8:]:
                tmp = np.append(tmp,abs(datadi[date]['PVRTOT'][idp[k],l]-datadi[date][pv][idp[k],l]))
            prodi[date][latlonid] = np.append(prodi[date][latlonid],np.where(tmp == np.min(tmp))[0][0])



gridmap['PVRRAD'] = gridmap['PVRSW'] + gridmap['PVRLWH'] + gridmap['PVRLWC']

nlab = np.array(['PVRCONVT','PVRCONVM','PVRTURBT','PVRTURBM','PVRRAD','PVRLS'])
process = np.zeros(counter.shape)
for k in range(len(counter[:,0])):
    for l in range(len(counter[0,:])):
        if counter[k,l]!=0:
            tmp = np.array([])
            for pv in nlab:
                tmp = np.append(tmp,abs(gridmap['PVRTOT'][k,l]-gridmap[pv][k,l]))
            process[k,l] = np.where(tmp==np.min(tmp))[0][0]+1


gridmap['counter']=counter
loc = np.where(counter!=0)


for pv in np.append(pvrs,'locres2'):
    gridmap[pv][loc]/=counter[loc]

loc=np.where(counter==0)
for pv in np.append(pvrs,'locres2'):
    gridmap[pv][loc]-=100

alpha=1.
linewidth=.2

minpltlatc = 15 
minpltlonc = -20

maxpltlatc = 60
maxpltlonc = 50

gridmap['PVRT'] = gridmap['PVRCONVT'] + gridmap['PVRTURBT']

pload = '/home/ascherrmann/010-IFS/data/DEC17/'
psave = '/home/ascherrmann/010-IFS/'
maxv = 3000
minv = 800
elv_levels = np.arange(minv,maxv,400)

data = xr.open_dataset(pload + 'IFSORO')

lon = data['lon']
lat = data['lat']
ZB = data['ZB'].values[0,0]

for pv in np.append(np.append(np.append('PVRT','counter'),pvrs),'locres2'):
    minv=-0.35
    maxv=0.35
    steps=0.05
    pvr_levels = np.arange(minv,maxv+0.0001,steps)
    ap = plt.cm.BrBG
    cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
    ticklabels=pvr_levels
    if pv=='PV':
        minv=-0.3
        maxv=1.5
        steps=0.1
        pvr_levels = np.arange(minv,maxv+0.0001,steps)
        cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
        ticklabels=pvr_levels
    if pv=='RES':
        minv=-2.0
        maxv=2.0
        steps=0.25
        pvr_levels = np.arange(minv,maxv+0.0001,steps)
        cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
        ticklabels=pvr_levels
    if pv=='counter':
        minv = 0
        maxv = 800
        steps= 20
        pvr_levels = np.arange(minv,maxv+0.0001,steps)
        cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
        ticklabels=pvr_levels
    if pv=='locres2':
        ap = plt.cm.cividis #PuBuGn
        minv = 0
        maxv = 0.5
        steps= 0.05
        pvr_levels = np.arange(minv,maxv+0.0001,steps)

        cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
        ticklabels=pvr_levels

    
    fig=plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(ncols=1, nrows=1)
    ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.8)
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')

    lc=ax.contourf(LON,LAT,gridmap[pv],levels=pvr_levels,cmap=cmap,extend='max',norm=norm)
#    lc.cmap.set_under('white')
    ax.contour(lon,lat,ZB,levels=elv_levels,linewidths=0.5,colors='black')
    if pv=='locres2':
        ax.contour(lon,lat,ZB,levels=elv_levels,linewidths=0.5,colors='purple')
    if pv=='RES' or pv=='locres' or pv=='locres2':
        lc.cmap.set_over('red')
    
    lonticks=np.arange(minpltlonc, maxpltlonc,5)
    latticks=np.arange(minpltlatc, maxpltlatc,5)
    
    ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
    ax.set_yticks(latticks, crs=ccrs.PlateCarree());
    ax.set_xticklabels(labels=lonticks,fontsize=8)
    ax.set_yticklabels(labels=latticks,fontsize=8)
    
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
    
    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(lc, ticks=pvr_levels,cax=cbax)
    
    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
    fig.canvas.mpl_connect('draw_event', func)
    
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel('PVU/h',fontsize=8)

    if pv=='PV' or pv=='RES' or pv=='locres' or pv=='locres2':
        cbar.ax.set_xlabel('PVU',fontsize=8)
    if pv=='DELTAPV':
        ax.text(-0.04,0.98,'a)',transform=ax.transAxes,fontweight='bold',fontsize=12)
    if pv=='PVRTOT':
        ax.text(-0.04,0.98,'b)',transform=ax.transAxes,fontweight='bold',fontsize=12)
    if pv=='locres2':
        ax.text(-0.04,0.98,'c)',transform=ax.transAxes,fontweight='bold',fontsize=12)

    if pv=='counter':
        cbar.ax.set_xlabel(' ',fontsize=8)

    cbar.ax.set_xticklabels(ticklabels)
    
    figname = psave + pv +'-gridmap' + '-PVedge-' + str(PVedge) + '.png'
    
    fig.savefig(figname,dpi=300,bbox_inches="tight")
    plt.close()

f = open(pload + 'process-di.txt','wb')
pickle.dump(prodi,f)
f.close()


domlabs = nlab
#cmap = ListedColormap(['orange','green','saddlebrown',,'dodgerblue','blue','magenta','salmon','red'])
cmap = ListedColormap(['orange','green','saddlebrown','dodgerblue','blue','red'])
#norm = BoundaryNorm([1 ,2, 3, 4, 5, 6,7,8,9], cmap.N)
norm = BoundaryNorm([1 ,2, 3, 4, 5, 6,7], cmap.N)
levels = np.arange(0.5,7,1)

ticklabels  = np.array([])

for q in domlabs:
    ticklabels = np.append(ticklabels,q[3:])

fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.coastlines()

lc=ax.contourf(LON,LAT,process,levels=levels,cmap=cmap,extend='both',norm=norm)
lc.cmap.set_under('white')
ax.contour(lon,lat,ZB,levels=elv_levels,linewidths=0.5,colors='black')

lonticks=np.arange(minpltlonc, maxpltlonc,5)
latticks=np.arange(minpltlatc, maxpltlatc,5)

ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
ax.set_yticks(latticks, crs=ccrs.PlateCarree());
ax.set_xticklabels(labels=lonticks,fontsize=8)
ax.set_yticklabels(labels=latticks,fontsize=8)

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())

cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(lc, ticks=levels,cax=cbax)

func=resize_colorbar_vert(cbax, ax, pad=0.01, size=0.02)
fig.canvas.mpl_connect('draw_event', func)

cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(' ',fontsize=8)
cbar.ax.set_yticklabels(ticklabels)

figname = psave + 'process-gridmap' + '-PVedge-' + str(PVedge) + '.png'
fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close()

