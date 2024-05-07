import numpy as np
import pandas as pd
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

import pickle

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
era5 = '/atmosdyn2/era5/cdf/'

cycmask = '/atmosdyn/michaesp/mincl.era-5/cdf.final/'
wcbmask = '/atmosdyn/katih/PhD/data/Gridding/grid_ERA5_r05_100_hit/'

f = open(ps + 'most-intense-weak-average-fields.txt','rb')
data = pickle.load(f)
f.close()

when = ['fourdaypriormature','fivedaypriormature','sixdaypriormature','sevendaypriormature','threedaypriormature','twodaypriormature','onedaypriormature','dates']

wl = ['4','5','6','7','3','2','1','0']

which = ['weak-cyclones.csv','intense-cyclones.csv']
minlon = -120
minlat = 10
maxlat = 80
maxlon = 80

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)
lonshort = np.linspace(-120,80,401)
latshort = np.linspace(10,80,141)
lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

# average cyclone 


cfl = np.arange(20,90.1,20)
seasons = ['DJF']#,'MAM','JJA','SON']
for sea in seasons:
    for wi in which[:]:
      sel = pd.read_csv(ps + sea + '-' + wi)

      #use the ll deepest cyclones
      for ll in [50, 100, 150, 200][-1:]:
        selp = sel.iloc[:ll]
        ### calc average
        for wq,we in zip(wl,when):
            TH850 = data[sea][wi][ll][we]['TH850']
            U300hPa = data[sea][wi][ll][we]['U300hPa']
            PV300hPa = data[sea][wi][ll][we]['PV300hPa']
            cycfreq = data[sea][wi][ll][we]['cycfreq']

            fig = plt.figure(figsize=(6,4))
            gs = gridspec.GridSpec(ncols=1, nrows=1)
            ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
            ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
            ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
            ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())
        
            cmap,pv_levels,norm,ticklabels=PV_cmap2()

            h = ax.contourf(LON[lons],LAT[lats],PV300hPa,cmap=cmap,levels=pv_levels,norm=norm,extend='both')

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

            name = 'season/PV-300hPa-no-con-for-' + wi[:-4] + '-' + wq + '-' + sea + '-' + '%d'%ll +'.png'
            fig.savefig(pi + name,dpi=300,bbox_inches='tight')
            plt.close('all')

            fig = plt.figure(figsize=(6,4))
            gs = gridspec.GridSpec(ncols=1, nrows=1)
            ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
            ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
            ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
            ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

            cmap=matplotlib.cm.gnuplot2
            ulvl = np.arange(-10,50.1,5)
            norm=BoundaryNorm(ulvl,cmap.N)

            h = ax.contourf(LON[lons],LAT[lats],U300hPa,cmap=cmap,levels=ulvl,norm=norm,extend='both')
            ax.contour(LON[lons],LAT[lats],cycfreq*100,linewidths=1,alpha=1,colors='mediumorchid',levels=cfl)
            lonticks=np.arange(minlon, maxlon+1,20)
            latticks=np.arange(minlat, maxlat+1,10)

            ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
            ax.set_yticks(latticks, crs=ccrs.PlateCarree());
            ax.set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
            ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)

            cbax = fig.add_axes([0, 0, 0.1, 0.1])
            cbar=plt.colorbar(h, ticks=ulvl,cax=cbax)
            func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
            fig.canvas.mpl_connect('draw_event', func)

            cbar.ax.tick_params(labelsize=10)
            cbar.ax.set_xlabel(r'm s$^{-1}$',fontsize=10)
            cbar.ax.set_xticklabels(ulvl)

            name = 'season/U-300hPa-no-con-for-' + wi[:-4] + '-' + wq + '-' + sea + '-' + '%d'%ll + '.png'
            fig.savefig(pi + name,dpi=300,bbox_inches='tight')
            plt.close('all')


            fig = plt.figure(figsize=(6,4))
            gs = gridspec.GridSpec(ncols=1, nrows=1)
            ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
            ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
            ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
            ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

            cmap=matplotlib.cm.seismic
            thlvl = np.arange(260,305,3)
            norm=BoundaryNorm(thlvl,cmap.N)

            h = ax.contourf(LON[lons],LAT[lats],TH850,cmap=cmap,levels=thlvl,norm=norm,extend='both')
            ax.contour(LON[lons],LAT[lats],cycfreq*100,linewidths=1,alpha=1,colors='mediumorchid',levels=cfl)
            lonticks=np.arange(minlon, maxlon+1,20)
            latticks=np.arange(minlat, maxlat+1,10)

            ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
            ax.set_yticks(latticks, crs=ccrs.PlateCarree());
            ax.set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
            ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)

            cbax = fig.add_axes([0, 0, 0.1, 0.1])
            cbar=plt.colorbar(h, ticks=thlvl[::2],cax=cbax)
            func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
            fig.canvas.mpl_connect('draw_event', func)

            cbar.ax.tick_params(labelsize=10)
            cbar.ax.set_xlabel(r'K',fontsize=10)
            cbar.ax.set_xticklabels(labels=thlvl[::2])

            name = 'season/TH-850hPa-no-con-for-' + wi[:-4] + '-' + wq + '-' + sea + '-' + '%d'%ll + '.png'
            fig.savefig(pi + name,dpi=300,bbox_inches='tight')
            plt.close('all')

