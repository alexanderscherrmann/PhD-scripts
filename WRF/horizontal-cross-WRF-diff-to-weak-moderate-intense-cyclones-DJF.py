import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from wrf import interplevel as intp
from netCDF4 import Dataset as ds
import netCDF4
import wrf
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec

import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
era5 = '/atmosdyn2/era5/cdf/'

when = ['fourdaypriormature','fivedaypriormature','sixdaypriormature','sevendaypriormature','threedaypriormature','twodaypriormature','onedaypriormature']
which = ['moderate-cyclones.csv']#,'weak-cyclones.csv','intense-cyclones.csv']
which = ['weak-cyclones.csv','intense-cyclones.csv']
minlon = -120
minlat = 10
maxlat = 79.5
maxlon = 79.5

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

### wrf reference data
data = netCDF4.Dataset('/atmosdyn2/ascherrmann/013-WRF-sim/015-mean-reference/wrfout_d01_2000-12-01_00:00:00')
PVwrf = wrf.getvar(data,'pvo',meta=False) #in PVU already
lonwrf = wrf.getvar(data,'lon',meta=False)[0]-0.25
latwrf = wrf.getvar(data,'lat',meta=False)[:,0]-0.25
Pwrf = wrf.getvar(data,'pressure',meta=False)
Uwrf = wrf.getvar(data,'U',meta=False)
Uwrf = Uwrf[:,:,1:]/2 + Uwrf[:,:,:-1]/2
THwrf = wrf.getvar(data,'T',meta=False)

THwrf_850hPa = intp(THwrf,Pwrf,850,meta=False) + 300
PVwrf_300hPa = intp(PVwrf,Pwrf,300.0,meta=False)
Uwrf_300hPa = intp(Uwrf,Pwrf,300.0,meta=False)

# average cyclone 

for wi in which:
    sel = pd.read_csv(ps + wi)
    mon = sel['month'].values
    
    ### calc average
    for we in when:
        PV300hPa  = np.zeros((len(lats),len(lons)))
        U300hPa = np.zeros((len(lats),len(lons)))
        TH850hPa = np.zeros((len(lats),len(lons)))
        for q,d in enumerate(sel[we].values):
          mo = mon[q]
          if mo==1 or mo==2 or mo==12:
            ep = era5 + d[:4] + '/' + d[4:6] + '/' 
            S = ds(ep + 'S' + d,mode='r')
            P = ds(ep + 'P' + d,mode='r')
#            PV = S.variables['PV'][0]
            PS = S.variables['PS'][0]
            TH = S.variables['TH'][0]
            U = P.variables['U'][0]
            hyam=P.variables['hyam']  # 137 levels  #für G-file ohne levels bis
            hybm=P.variables['hybm']  #   ''
            ak=hyam[hyam.shape[0]-98:] # only 98 levs are used:
            bk=hybm[hybm.shape[0]-98:]
            
            ps3d=np.tile(PS[:,:],(len(ak),1,1))
            Pr=(ak/100.+bk*ps3d.T).T

            u300hpa = intp(U,Pr,300,meta=False)
            th850hpa = intp(TH,Pr,850,meta=False)
#            pv300hpa = intp(PV,Pr,300,meta=False)
#            PV300hPa += pv300hpa[la0:la1,lo0:lo1]
            U300hPa += u300hpa[la0:la1,lo0:lo1]
            TH850hPa += th850hpa[la0:la1,lo0:lo1]

#        PV300hPa/=len(np.where((mon==1)|(mon==2)|(mon==12))[0])
        U300hPa/=len(np.where((mon==1)|(mon==2)|(mon==12))[0])
        TH850hPa/=len(np.where((mon==1)|(mon==2)|(mon==12))[0])

        fig = plt.figure(figsize=(6,4))
        gs = gridspec.GridSpec(ncols=1, nrows=1)
        ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
        ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

        cmap=matplotlib.cm.BrBG
        ulvl = np.arange(-15,15.1,3)
        norm=BoundaryNorm(ulvl,cmap.N)

        h = ax.contourf(LON[lons],LAT[lats],U300hPa-Uwrf_300hPa,cmap=cmap,levels=ulvl,norm=norm,extend='both')
        lonticks=np.arange(minlon, maxlon+1,20)
        latticks=np.arange(minlat, maxlat+1,10)

        ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
        ax.set_yticks(latticks, crs=ccrs.PlateCarree());
        ax.set_xticklabels(labels=lonticks[:-1],fontsize=10)
        ax.set_yticklabels(labels=latticks,fontsize=10)

        cbax = fig.add_axes([0, 0, 0.1, 0.1])
        cbar=plt.colorbar(h, ticks=ulvl,cax=cbax)
        func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
        fig.canvas.mpl_connect('draw_event', func)

        cbar.ax.tick_params(labelsize=10)
        cbar.ax.set_xlabel(r'm s$^{-1}$',fontsize=10)
        cbar.ax.set_xticklabels(ulvl)

        name = 'U-diff-300hPa-for-' + wi[:-4] + '-' + we + '-DJF.png'
        fig.savefig(pi + name,dpi=300,bbox_inches='tight')
        plt.close('all')


        fig = plt.figure(figsize=(6,4))
        gs = gridspec.GridSpec(ncols=1, nrows=1)
        ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
        ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

        cmap=matplotlib.cm.seismic
        thlvl = np.arange(-6,6.1,1)
        norm=BoundaryNorm(thlvl,cmap.N)

        h = ax.contourf(LON[lons],LAT[lats],TH850hPa-THwrf_850hPa,cmap=cmap,levels=thlvl,norm=norm,extend='both')
        lonticks=np.arange(minlon, maxlon+1,20)
        latticks=np.arange(minlat, maxlat+1,10)

        ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
        ax.set_yticks(latticks, crs=ccrs.PlateCarree());
        ax.set_xticklabels(labels=lonticks[:-1],fontsize=10)
        ax.set_yticklabels(labels=latticks,fontsize=10)

        cbax = fig.add_axes([0, 0, 0.1, 0.1])
        cbar=plt.colorbar(h, ticks=thlvl,cax=cbax)
        func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
        fig.canvas.mpl_connect('draw_event', func)

        cbar.ax.tick_params(labelsize=10)
        cbar.ax.set_xlabel(r'K',fontsize=10)
        cbar.ax.set_xticklabels(thlvl)

        name = 'TH-diff-850hPa-for-' + wi[:-4] + '-' + we + '-DJF.png'
        fig.savefig(pi + name,dpi=300,bbox_inches='tight')
        plt.close('all')
