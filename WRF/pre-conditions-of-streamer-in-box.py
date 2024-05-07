import pickle
import numpy as np

f = open('/atmosdyn2/ascherrmann/013-WRF-sim/data/streamer-in-boxes.txt','rb')
da = pickle.load(f)
f.close()

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from wrf import interplevel as intp
from netCDF4 import Dataset as ds

import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec

import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert


def get_Xdays_before_mature_stage(d,X):
    if (int(d[6:8])-X)>=1:
        return d[:6] + '%02d'%(int(d[6:8])-X) + d[-3:]

    monthsn = int(d[4:6])-1
    if monthsn<1:
        return '%d'%(int(d[:4])-1) + '12' + '%02d'%(int(d[6:8]) + 31 - X) + d[-3:]

    if monthsn<8 and monthsn%2==1:
        days = 31
    elif monthsn==2:
        days =28
        if int(d[:4])%4==0:
            days=29
    elif monthsn>=8 and monthsn%2==0:
        days=31
    else:
        days=30
    return d[:4] + '%02d'%(monthsn) + '%02d'%(int(d[6:8]) + days - X) + d[-3:]


LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

lon = np.linspace(-120,80,401)
lat = np.linspace(10,80,141)

minlon = -120
minlat = 10
maxlat = 80
maxlon = 80

lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
cmap,pv_levels,norm,ticklabels=PV_cmap2()

k='7.537.5'

era5 = '/atmosdyn/era5/cdf/'
for x in range(3,7):
    djf = np.zeros((len(lat),len(lon)))
    counter=0
    for d in da[k]:
        if d=='n':
            continue
        if int(d[4:6])==12 or int(d[4:6])==1 or int(d[4:6])==2:
            dtmp = get_Xdays_before_mature_stage(d,x)
            f = era5 + dtmp[:4] + '/' + dtmp[4:6] + '/S' + dtmp
            S = ds(f,mode='r')
            PV = S.variables['PV'][0,:,la0:la1,lo0:lo1]
            TH = S.variables['TH'][0,:,la0:la1,lo0:lo1]
            PS = S.variables['PS'][0,la0:la1,lo0:lo1]
            hyam=S.variables['hyam']  # 137 levels  #für G-file ohne levels bis
            hybm=S.variables['hybm']  #   ''
            ak=hyam[hyam.shape[0]-98:] # only 98 levs are used:
            bk=hybm[hybm.shape[0]-98:]
            ps3d=np.tile(PS[:,:],(len(ak),1,1))
            Pr=(ak/100.+bk*ps3d.T).T

            PV300 = intp(PV,Pr,300,meta=False)
            djf+=PV300

            counter+=1

    djf/=counter

    fig = plt.figure(figsize=(6,4))
    gs = gridspec.GridSpec(ncols=1, nrows=1)
    ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
    ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

    h = ax.contourf(lon,lat,djf,cmap=cmap,levels=pv_levels,norm=norm,extend='both')
    lonticks=np.arange(minlon, maxlon+1,20)
    latticks=np.arange(minlat, maxlat+1,10)

    ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
    ax.set_yticks(latticks, crs=ccrs.PlateCarree());
    ax.set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
    ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)

    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(h, ticks=pv_levels,cax=cbax)
    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
    fig.canvas.mpl_connect('draw_event', func)

    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_xlabel('PVU',fontsize=10)
    cbar.ax.set_xticklabels(ticklabels)

    name = 'PV-300hPa-%d-days-prior-pv-in'%x + k + '-box.png'
    fig.savefig(pi + name,dpi=300,bbox_inches='tight')
    plt.close('all')

