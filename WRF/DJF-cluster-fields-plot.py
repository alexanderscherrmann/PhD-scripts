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

f = open(ps + 'DJF-individual-fields.txt','rb')
data = pickle.load(f)
f.close()

when = ['fourdaypriormature','fivedaypriormature','sixdaypriormature','sevendaypriormature','threedaypriormature','twodaypriormature','onedaypriormature','dates']

wl = ['4','5','6','7','3','2','1','0']

which = ['intense-cyclones.csv']
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
seasons = ['DJF','MAM','JJA','SON']
plotvar = ['TH850','U300hPa','omega500','PV300hPa','MSL']

cmap_pv,pv_levels,norm_pv,ticklabels_pv=PV_cmap2()

units = ['K','m s$^{-1}$','Pa s$^{-1}$','PVU','hPa']
cmaps = [matplotlib.cm.seismic,matplotlib.cm.gnuplot2,matplotlib.cm.BrBG,cmap_pv,matplotlib.cm.cividis]
levels = [np.arange(260,305,3),np.arange(-10,50.1,5),np.arange(-0.5,0.6,0.1),pv_levels,np.arange(990,1031,5)]
norms = [BoundaryNorm(levels[0],cmaps[0].N),BoundaryNorm(levels[1],cmaps[1].N),BoundaryNorm(levels[2],cmaps[2].N),norm_pv,BoundaryNorm(levels[4],cmaps[4].N)]
names = ['TH-850hPa','U-300hPa','omega-500hPa','PV-300hPa','MSL']

for sea in seasons[:1]:
    pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/season/'
    pi += sea +'/clusters/'
    if not os.path.isdir(pi):
        os.mkdir(pi)
    for plv in plotvar:
        if not os.path.isdir(pi + plv + '/'):
            os.mkdir(pi + plv)

    for wi in which:
      sel = pd.read_csv(ps + sea + '-' + wi)

      #use the ll deepest cyclones
      for ll in [200]:
        sel = sel.iloc[:ll]

        for clus in np.unique(sel['region'].values):
            ids = sel['ID'].values[np.where(sel['region'].values==clus)[0]]
            c = len(ids)
            for wq,we in zip(wl,when):
                plots = dict()
                for plv in plotvar:
                    plots[plv] = np.zeros((141,401))

                for i in ids:
                ### calc average
                    for plv in plotvar:
                        plots[plv]+= data[sea][wi][ll][we][i][plv]/c

                plots['MSL']/=100
                for un,lvl,nor,cmap,var,nam in zip(units,levels,norms,cmaps,plotvar,names):
                 if var=='MSL':
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
            
                    name = nam + '-for-' + clus + '-' + wq + '-%d'%c +'.png'
                    fig.savefig(pi + var + '/' + name,dpi=300,bbox_inches='tight')
                    plt.close('all')


