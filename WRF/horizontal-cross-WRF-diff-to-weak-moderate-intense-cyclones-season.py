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
import pickle
import sys
import os
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
era5 = '/atmosdyn2/era5/cdf/'

when = ['fourdaypriormature','fivedaypriormature','sixdaypriormature','sevendaypriormature','threedaypriormature','twodaypriormature','onedaypriormature','dates']
wl = ['4','5','6','7','3','2','1','0']


which = ['weak-cyclones.csv','moderate-cyclones.csv','intense-cyclones.csv']

f = open(ps + 'new-most-intense-moderate-weak-average-fields.txt','rb')
dat = pickle.load(f)
f.close()

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

seasons = ['DJF','MAM','JJA','SON']
plotvar = ['U-diff-300','TH-diff-850','omega-diff-500']

for sea in seasons:
    pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/season/'
    pi += sea +'/'
    if not os.path.isdir(pi):
        os.mkdir(pi)
    for plv in plotvar:
        if not os.path.isdir(pi + plv + '/'):
            os.mkdir(pi + plv)

    data = netCDF4.Dataset('/atmosdyn2/ascherrmann/013-WRF-sim/' + sea + '-clim/wrfout_d01_2000-12-01_00:00:00')
    PVwrf = wrf.getvar(data,'pvo',meta=False) #in PVU already
    lonwrf = wrf.getvar(data,'lon',meta=False)[0]-0.25
    latwrf = wrf.getvar(data,'lat',meta=False)[:,0]-0.25
    Pwrf = wrf.getvar(data,'pressure',meta=False)
    Uwrf = wrf.getvar(data,'U',meta=False)
    Uwrf = Uwrf[:,:,1:]/2 + Uwrf[:,:,:-1]/2
    THwrf = wrf.getvar(data,'T',meta=False)
    omegawrf = wrf.getvar(data,'omega',meta=False)

    THwrf_850hPa = intp(THwrf,Pwrf,850,meta=False) + 300
    PVwrf_300hPa = intp(PVwrf,Pwrf,300.0,meta=False)
    Uwrf_300hPa = intp(Uwrf,Pwrf,300.0,meta=False)
    omegawrf_500hPa = intp(omegawrf,Pwrf,500,meta=False)

    for wi in which:
      sel = pd.read_csv(ps + sea + '-' + wi)
      #use the ll deepest cyclones
      for ll in [50, 100, 150, 200]:
          for plv in plotvar:
            if not os.path.isdir(pi + plv + '/' + '%d'%ll):
                os.mkdir(pi + plv + '/' + '%d'%ll)

        ### calc average
          for wq,we in zip(wl,when): 
            U300hPa = dat[sea][wi][ll][we]['U300hPa'][:-1,:-1]
            TH850hPa = dat[sea][wi][ll][we]['TH850'][:-1,:-1]
            omega500hPa = dat[sea][wi][ll][we]['omega500'][:-1,:-1]

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
            ax.set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
            ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)
    
            cbax = fig.add_axes([0, 0, 0.1, 0.1])
            cbar=plt.colorbar(h, ticks=ulvl,cax=cbax)
            func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
            fig.canvas.mpl_connect('draw_event', func)
    
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.set_xlabel(r'm s$^{-1}$',fontsize=10)
            cbar.ax.set_xticklabels(ulvl)
    
            name = 'U-diff-300hPa-for-' + wi[:-4] + '-' + wq + '-' + sea + '-' + '%03d'%ll +'.png'
            fig.savefig(pi + plotvar[0] + '/%d/'%ll + name,dpi=300,bbox_inches='tight')
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
            ax.set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
            ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)
    
            cbax = fig.add_axes([0, 0, 0.1, 0.1])
            cbar=plt.colorbar(h, ticks=thlvl,cax=cbax)
            func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
            fig.canvas.mpl_connect('draw_event', func)
    
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.set_xlabel(r'K',fontsize=10)
            cbar.ax.set_xticklabels(thlvl)
    
            name = 'TH-diff-850hPa-for-' + wi[:-4] + '-' + wq + '-' + sea + '-' + '%03d'%ll +'.png'
            fig.savefig(pi + plotvar[1] + '/%d/'%ll +  name,dpi=300,bbox_inches='tight')
            plt.close('all')


            fig = plt.figure(figsize=(6,4))
            gs = gridspec.GridSpec(ncols=1, nrows=1)
            ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
            ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
            ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
            ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

            cmap=matplotlib.cm.PiYG
            omegalvl = np.arange(-1,1.1,0.1)
            norm=BoundaryNorm(omegalvl,cmap.N)

            h = ax.contourf(LON[lons],LAT[lats],omega500hPa-omegawrf_500hPa,cmap=cmap,levels=omegalvl,norm=norm,extend='both')
            lonticks=np.arange(minlon, maxlon+1,20)
            latticks=np.arange(minlat, maxlat+1,10)

            ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
            ax.set_yticks(latticks, crs=ccrs.PlateCarree());
            ax.set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
            ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)

            cbax = fig.add_axes([0, 0, 0.1, 0.1])
            cbar=plt.colorbar(h, ticks=omegalvl,cax=cbax)
            func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
            fig.canvas.mpl_connect('draw_event', func)

            cbar.ax.tick_params(labelsize=10)
            cbar.ax.set_xlabel(r'Pa $s^{-1}$',fontsize=10)
            cbar.ax.set_xticklabels(omegalvl)

            name = 'omega-diff-500hPa-for-' + wi[:-4] + '-' + wq + '-' + sea + '-' + '%03d'%ll +'.png'
            fig.savefig(pi + plotvar[2] + '/%d/'%ll + name,dpi=300,bbox_inches='tight')
            plt.close('all')
