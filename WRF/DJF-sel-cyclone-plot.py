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
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
era5 = '/atmosdyn2/era5/cdf/'

f = open(ps + 'DJF-intense-selected-PV-2.txt','rb')
data = pickle.load(f)
f.close()

when = ['fourdaypriormature','fivedaypriormature','sixdaypriormature','sevendaypriormature','threedaypriormature','twodaypriormature','onedaypriormature','dates']

wl = ['4','5','6','7','3','2','1','0']

which = ['weak-cyclones.csv','moderate-cyclones.csv','intense-cyclones.csv']
minlon = -120
minlat = 10
maxlat = 80
maxlon = 80

minlon2 = -10
maxlon2 = 45
minlat2 = 25
maxlat2 = 50

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)
lonshort = np.linspace(-120,80,401)
latshort = np.linspace(10,80,141)
lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

# average cyclone 


cfl = np.arange(20,90.1,20)
seasons = ['DJF','MAM','JJA','SON']
plotvar = ['TH850','U300hPa','omega500','PV300hPa','Precip24','MSL']

cmap_pv,pv_levels,norm_pv,ticklabels_pv=PV_cmap2()

units = ['K','m s$^{-1}$','Pa s$^{-1}$','PVU','mm','hPa']
cmaps = [matplotlib.cm.seismic,matplotlib.cm.gnuplot2,matplotlib.cm.BrBG,cmap_pv,matplotlib.cm.YlGnBu,matplotlib.cm.cividis]
levels = [np.arange(260,305,3),np.arange(-10,50.1,5),np.arange(-0.5,0.6,0.1),pv_levels,np.arange(0,11,1),np.arange(990,1031,5)]
norms = [BoundaryNorm(levels[0],cmaps[0].N),BoundaryNorm(levels[1],cmaps[1].N),BoundaryNorm(levels[2],cmaps[2].N),norm_pv,BoundaryNorm(levels[4],cmaps[4].N),BoundaryNorm(levels[5],cmaps[5].N)]
names = ['TH-850hPa','U-300hPa','omega-500hPa','PV-300hPa','Precip24','MSL']
color = ['b','k','r']
for sea in seasons[:1]:
    pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/season/'
    pi += sea +'/sel-cyclones/'
    if not os.path.isdir(pi):
        os.mkdir(pi)
    for plv in plotvar:
        if not os.path.isdir(pi + plv + '/'):
            os.mkdir(pi + plv)

    for co,wi in zip(color[2:],which[2:]):
      sel = pd.read_csv(ps + sea + '-' + wi)

      #use the ll deepest cyclones
      for ll in [50]:#, 100, 150, 200]:
        for plv in plotvar:
            if not os.path.isdir(pi + plv + '/' + '%d'%ll):
                os.mkdir(pi + plv + '/' + '%d'%ll)

        selp = sel.iloc[:ll]
        if co=='r':
            selp = pd.read_csv('/atmosdyn2/ascherrmann/013-WRF-sim/data/selected-intense-cyclones.csv')
        ### calc average
        plots = dict()
        for wq,we in zip(wl,when):
            plots['PV300hPa'] = data[sea][wi][ll][we]['PV300hPa']

            for un,lvl,nor,cmap,var,nam in zip(units,levels,norms,cmaps,plotvar,names):
             if var=='PV300hPa':
                fig = plt.figure(figsize=(6,4))
                gs = gridspec.GridSpec(ncols=1, nrows=1)
                ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
                ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
                ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
                ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())
            
                h = ax.contourf(LON[lons],LAT[lats],plots[var],cmap=cmap,levels=lvl,norm=nor,extend='both')

                lonticks=np.arange(minlon, maxlon+1,20)
                latticks=np.arange(minlat, maxlat+1,10)
    
                ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
                ax.set_yticks(latticks, crs=ccrs.PlateCarree());
                ax.set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
                ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)
    
                cbax = fig.add_axes([0, 0, 0.1, 0.1])
                cbar=plt.colorbar(h, ticks=lvl,cax=cbax)
                func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
                fig.canvas.mpl_connect('draw_event', func)
    
                cbar.ax.tick_params(labelsize=10)
                cbar.ax.set_xlabel(un,fontsize=10)
                cbar.ax.set_xticklabels(lvl)
    
                name = nam + '-for-' + wi[:-4] + '-' + wq + '-' + sea + '-' + '%d'%len(selp['lon'].values) +'.png'
                fig.savefig(pi + var + '/%d/'%ll + name,dpi=300,bbox_inches='tight')
                plt.close('all')


selp = pd.read_csv('/atmosdyn2/ascherrmann/013-WRF-sim/data/selected-intense-cyclones.csv')

fig = plt.figure(figsize=(6,4))
gs = gridspec.GridSpec(ncols=1, nrows=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')

ax.scatter(selp['startlon'].values,selp['startlat'].values,color=co,marker='d')
ax.scatter(selp['lon'].values,selp['lat'].values,color=co,marker='.')

ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
ax.set_yticks(latticks, crs=ccrs.PlateCarree());
ax.set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)
ax.set_extent([minlon2, maxlon2, minlat2, maxlat2], ccrs.PlateCarree())

name = 'start-mature-pos-for-' + wi[:-4] + '-' + sea + '-' + '%d'%ll +'.png'
name = 'start-mature-pos-for-' + wi[:-4] + '-' + sea + '-' + '%d'%len(selp['lon'].values) +'.png'

if not os.path.isdir(pi + '/%d/'%ll):
    os.mkdir(pi + '/%d/'%ll)

fig.savefig(pi + '/%d/'%ll + name,dpi=300,bbox_inches='tight')
plt.close('all')


fig,axes = plt.subplots(1,2)
axes = axes.flatten()
ax = axes[0]
ax.hist(selp['lifetime'].values-selp['htSLPmin'].values,bins=32,range=[0,48],edgecolor='k',facecolor='grey')
ax.set_xlabel('lifetime after mature [h]')

ax = axes[1]
ax.hist(selp['startslp'].values-selp['minSLP'].values,bins=32,range=[0,32],edgecolor='k',facecolor='grey')
ax.set_xlabel('$\Delta$ SLP [hPa]')

name = 'diff-in-slp-and-age-for-' + wi[:-4] + '-' + sea + '-' + '%d'%ll +'.png'
name = 'diff-in-slp-and-age-for-' + wi[:-4] + '-' + sea + '-' + '%d'%len(selp['lon'].values) +'.png'
fig.savefig(pi + '/%d/'%ll + name,dpi=300,bbox_inches='tight')
plt.close('all')



