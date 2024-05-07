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

pload = '/home/ascherrmann/010-IFS/ctraj/' + CT + '/use/'

f = open(pload + 'PV-data-'+CT+'dPSP-100-ZB-800PVedge-0.3-400-correct-distance.txt','rb')
data = pickle.load(f)
f.close()

f = open('/home/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','rb')
ldata = pickle.load(f)
f.close()

NORO = xr.open_dataset('/home/ascherrmann/010-IFS/data/IFSORO')
ZB = NORO['ZB'].values[0,0]
oro = data['oro']
datadi = data['rawdata']
dipv = data['dipv']

rdis = 400
labs = helper.traced_vars_IFS()
H = 48
a = 1

maxv = 0.61
minv =-0.6
pvr_levels = np.arange(minv,maxv,0.15)

ap = plt.cm.seismic
cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
for k in range(75,150):
    cmap.colors[k] = np.array([189/256, 195/256, 199/256, 1.0])

LON = np.arange(-180,180.1,0.4)
LAT = np.arange(0,90.1,0.4)

alpha=1.
linewidth=1
ticklabels=pvr_levels
text = ['a)','b)']
pvsum = np.where(labs=='PVRCONVT')[0][0]
for q,date in enumerate(oro.keys()):
# if (datadi[date]['highORO']==1):
 if(date=='20171214_02-073'):# or date=='20180619_03-111':
  mon = data['mons'][q]
  CYID = int(date[-3:])
  idp = np.where(datadi[date]['PV'][:,0]>=0.75)[0]
  if(np.mean(datadi[date]['OL'][idp,0])<0.9):

    tralon = datadi[date]['lon'][idp,:]
    tralat = datadi[date]['lat'][idp,:]

    PVoro = oro[date]['env']['APVTOT'][idp,:]
    deltaPVoro = np.zeros(datadi[date]['time'][idp,:].shape)
    deltaPVoro[:,1:] = PVoro[:,:-1]-PVoro[:,1:]

    pvr = deltaPVoro
    pvr = np.zeros(datadi[date]['time'][idp,:].shape)

    for k in labs[pvsum:]:
        pvr += datadi[date][k][idp,:]
#    pvr[:,1:] = (dipv[date]['cyc']['APVTOT'][idp,:-1] + dipv[date]['env']['APVTOT'][idp,:-1] -
#            dipv[date]['cyc']['APVTOT'][idp,1:] + dipv[date]['env']['APVTOT'][idp,1:])

    tracklo = np.array([])
    trackla = np.array([])
    for u in range(len(ldata[mon][CYID]['clon'])):
        tracklo = np.append(tracklo,np.mean(LON[ldata[mon][CYID]['clon'][u].astype(int)]))
        trackla = np.append(trackla,np.mean(LAT[ldata[mon][CYID]['clat'][u].astype(int)]))
        if ldata[mon][CYID]['dates'][u]==date[:-4]:
            latc = np.mean(LAT[ldata[mon][CYID]['clat'][u].astype(int)])
            lonc = np.mean(LON[ldata[mon][CYID]['clon'][u].astype(int)])
            matureid = u

#    latc = np.mean(np.unique(tralat[:,0]))
#    lonc = np.mean(np.unique(tralon[:,0]))

    if date=='20171214_02-073':
        minpltlatc = np.round(latc-np.floor(helper.convert_radial_distance_to_lon_lat_dis(1200)),0)
        minpltlonc = np.round(lonc-np.floor(helper.convert_radial_distance_to_lon_lat_dis(2200)),0)

        maxpltlatc = np.round(latc+np.round(helper.convert_radial_distance_to_lon_lat_dis(2000),0),0)
        maxpltlonc = np.round(lonc+np.round(helper.convert_radial_distance_to_lon_lat_dis(1000),0),0)
    else:
        minpltlatc = np.round(latc-np.floor(helper.convert_radial_distance_to_lon_lat_dis(1600)),0)
        minpltlonc = np.round(lonc-np.floor(helper.convert_radial_distance_to_lon_lat_dis(2200)),0)

        maxpltlatc = np.round(latc+np.round(helper.convert_radial_distance_to_lon_lat_dis(1600),0),0)
        maxpltlonc = np.round(lonc+np.round(helper.convert_radial_distance_to_lon_lat_dis(1600),0),0)


#    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
#    ax.coastlines()

    fig=plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(ncols=1, nrows=1)
    ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)

    for q in range(len(tralon[:,0])):
        seg = helper.make_segments(tralon[q,matureid-1:matureid+1],tralat[q,matureid-1:matureid+1])
        z = pvr[q,matureid-1:matureid+1]
        lc = mcoll.LineCollection(seg, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
        ax=plt.gca()
        ax.add_collection(lc)

    lonticks=np.arange(minpltlonc, maxpltlonc,5)
    latticks=np.arange(minpltlatc, maxpltlatc+1,5)

    ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
    ax.set_yticks(latticks, crs=ccrs.PlateCarree());
    ax.set_xticklabels(labels=lonticks,fontsize=8)
    ax.set_yticklabels(labels=latticks,fontsize=8)
    
    ax.contour(LON,LAT,ZB,levels=np.arange(800,3000,400),linewidths=0.5,colors='k')
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
    ax.plot(tracklo,trackla,color='k')
    ax.scatter(tracklo[0],trackla[0],color='k',marker='o',s=25,zorder=50)
#    ax.scatter(tralon[idp,20],tralat[idp,20],color='k',marker='s',s=10,zorder=50)
#    ax.scatter(lonc,latc,color='k',marker='o',s=25,zorder=50)
    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(lc, ticks=pvr_levels,cax=cbax)

    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
    fig.canvas.mpl_connect('draw_event', func)

    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel('PVU/h',fontsize=8)
    cbar.ax.set_xticklabels(ticklabels)

    circ2 = patch.Circle((tracklo[0],trackla[0]),helper.convert_radial_distance_to_lon_lat_dis(400),edgecolor='k',facecolor='none',linestyle='-',linewidth=1.5,zorder=10.)
#    circ1 = patch.Circle((lonc,latc),helper.convert_radial_distance_to_lon_lat_dis(200),edgecolor='k',linestyle='-',facecolor='None',linewidth=.5,zorder=10.)

#    ax.add_patch(circ1)
    ax.add_patch(circ2)
    if(date=='20171214_02-073'):
        te = text[0]
    else:
        te = text[1]
    ax.text(0.05, 0.95, te, transform=ax.transAxes,fontsize=12, fontweight='bold', va='top')
    figname = pload + 'traj-pvr-' + str(date)  + '-interview3.png'
    fig.savefig(figname,dpi=300,bbox_inches="tight")
    plt.close()


