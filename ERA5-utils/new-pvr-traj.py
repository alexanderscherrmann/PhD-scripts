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

pload = '/home/ascherrmann/009-ERA-5/MED/traj/use/'

df = pd.read_csv(pload[:-4] + 'pandas-all-data.csv')
df = df.loc[df['ntraj075']>=200]

ID = df['ID'].values
lon = df['lon'].values
lat = df['lat'].values
SLP = df['minSLP'].values
hourstoSLPmin = df['htminSLP'].values
ntraj075 = df['ntraj075'].values
PVano = df['ano'].values
envano = df['envperano'].values
advano = df['advperano'].values


savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']
var = []

for u,x in enumerate(savings):
    f = open(pload[:-4] + x + 'furthersel.txt',"rb")
    var.append(pickle.load(f))
    f.close()

tlon = var[1]
tlat = var[2]
aID = var[3]

avaID = np.array([])
for k in range(len(aID)):
    avaID=np.append(avaID,aID[k][0].astype(int))

f = open(pload + 'PV-data-dPSP-100-ZB-800-2-400.txt','rb')
data = pickle.load(f)
f.close()

NORO = xr.open_dataset('/home/ascherrmann/scripts/ERA5-utils/NORO')
ZB = NORO['ZB'].values[0]
Zlon = NORO['lon']
Zlat = NORO['lat']
Emaxv = 3000
Eminv = 800
elevation_levels = np.arange(Eminv,Emaxv,400)

oro = data['oro']
datadi = data['rawdata']

rdis = 400
H = 48
a = 1


maxv = 0.61
minv =-0.6
#maxv = 0.1
#minv= -48
pvr_levels = np.arange(minv,maxv,0.15)
#pvr_levels = np.arange(minv,maxv,1)

ap = plt.cm.seismic
#ap = plt.cm.nipy_spectral
cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels),pvr_levels)
for k in range(75,150):
    cmap.colors[k] = np.array([189/256, 195/256, 199/256, 1.0])

alpha=1.
linewidth=.2
ticklabels=pvr_levels

labs = helper.traced_vars_ERA5MED()

#datei = np.array(['333838','4725','455990','183340','108215','319708','524225','390733','313864','291127','51160','146370','392579','13867','150987','320710','449217','533339','339524','119896'])
# ID lon lat minSLP htminSLP ntraj075 PVanomaly envperano advperano

datei=np.array(['119896'])#,'108215'])#,'449217','460693','529090'])
datei=np.array(['108215'])

save = np.zeros((len(datei),9))
co = 0
for q,date in enumerate(datadi.keys()):
  if np.all(datei!=date):
      continue
  
  fig = plt.figure(figsize=(6,4))
  gs = gridspec.GridSpec(nrows=1,ncols=1)
  ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
  ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)

  ids = np.where(ID==int(date))[0][0]
  I = np.where(avaID==int(date))[0][0]
  idp = np.where(datadi[date]['PV'][:,0]>=0.75)[0]
  if True:

    save[co,0] = int(date)
    save[co,1] = lon[ids]
    save[co,2] = lat[ids]
    save[co,3] = SLP[ids]
    save[co,4] = hourstoSLPmin[ids]
    save[co,5] = ntraj075[ids]
    save[co,6] = PVano[ids]
    save[co,7] = envano[ids]
    save[co,8] = advano[ids]

    tralon = datadi[date]['lon'][idp,:] 
    tralat = datadi[date]['lat'][idp,:]

    deltaPV = np.zeros(datadi[date]['time'][idp,:].shape)
    deltaPV[:,1:] = datadi[date]['PV'][idp,:-1]-datadi[date]['PV'][idp,1:]

    PVoro = oro[date]['env'][idp,:]
    deltaPVoro = np.zeros(datadi[date]['time'][idp,:].shape)
    deltaPVoro[:,1:] = PVoro[:,:-1]-PVoro[:,1:]
    pvr = deltaPV   

    latc = lat[ids]
    lonc = lon[ids]   

    minpltlatc = np.round(latc-np.floor(helper.convert_radial_distance_to_lon_lat_dis(2000)),0)
    minpltlonc = np.round(lonc-np.floor(helper.convert_radial_distance_to_lon_lat_dis(2000)),0)

    maxpltlatc = np.round(latc+np.round(helper.convert_radial_distance_to_lon_lat_dis(1000),0),0)
    maxpltlonc = np.round(lonc+np.round(helper.convert_radial_distance_to_lon_lat_dis(2000),0),0)

    
    
    ax.contour(Zlon,Zlat,ZB,levels = elevation_levels,colors='purple',linewidths=0.35,alpha=1)

    for q in range(len(tralon[:,0])):
        seg = helper.make_segments(tralon[q,:],tralat[q,:])
        z = pvr[q,:]
        lc = mcoll.LineCollection(seg, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
        ax=plt.gca()
        ax.add_collection(lc)
    ax.plot(tlon[I],tlat[I],color='k')
#    ax.scatter(tlon[I][0],tlat[I][0],marker='o',color='k',s=20,zorder=10)
    ax.scatter(tlon[I][14],tlat[I][14],marker='o',color='k',s=20,zorder=10)
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
    cbar.ax.set_xticklabels(ticklabels)
    ax.text(-0.1,0.9,'a)',transform=ax.transAxes,fontweight='bold',fontsize=12)
#    circ2 = patch.Circle((lonc,latc),helper.convert_radial_distance_to_lon_lat_dis(400),edgecolor='k',facecolor='none',linestyle='--',linewidth=.5,zorder=10.)
#    circ1 = patch.Circle((lon[ids][0],lat[ids][0]),helper.convert_radial_distance_to_lon_lat_dis(400),edgecolor='k',linestyle='-',facecolor='None',linewidth=.5,zorder=10.)

#    ax.add_patch(circ1)
#    ax.add_patch(circ2)    

    figname = pload[:-9] + 'traj-pvr-' + str(date)  + '.png'
    fig.savefig(figname,dpi=300,bbox_inches="tight")
    plt.close('all')
    co+=1

#np.savetxt(pload[:-9] + 'orocases-data.txt',save,fmt='%.3f',delimiter=' ',newline='\n')

