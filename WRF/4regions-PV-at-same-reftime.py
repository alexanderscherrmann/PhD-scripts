from netCDF4 import Dataset as ds
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec
import matplotlib

import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert

minlon = -80
minlat = 30
maxlat = 85
maxlon = 60

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/4regionsPV/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/consec-avPV-field/'
era5 = '/atmosdyn2/era5/cdf/'

f = open(ps + '100-region-season.txt','rb')
d = pickle.load(f)
f.close()

cmap,levels,norm,ticklabels=PV_cmap2()
levels = np.arange(-3,3.5,0.5)
for sea in ['DJF','SON']:
    for re in ['east','west','central','black']:
        if not os.path.isdir(pi):
            os.mkdir(pi)
        if not os.path.isdir(pi+sea):
            os.mkdir(pi+sea)
        if not os.path.isdir(pi + sea + '/'+re):
            os.mkdir(pi + sea + '/' + re)
        pii = pi + sea +'/' +re +'/'

        for k in range(0,65,3):
                
            fig = plt.figure(figsize=(6,4))
            gs = gridspec.GridSpec(ncols=1, nrows=1)
            ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
            ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
            ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
            ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())
            counters = np.zeros((len(lats),len(lons)))
            PV = np.zeros_like(counters)
            MSL = np.zeros_like(counters)
            
            for dirs in d[sea][re]:
                dirs = '%06d'%dirs
                dates = np.array([])
                for f in os.listdir(ps + dirs + '/300/'):
                    if f.startswith('D'):
                        dates = np.append(dates,f)
        
                date = dates[k]
                S=ds(ps + dirs + '/300/' + date,'r')
                PV+=S.variables['PV'][0,0,la0:la1,lo0:lo1]
                #PVr[rm==rv]+=S.variables['PV'][0,0,la0:la1,lo0:lo1][rm==rv]
                B = ds(era5 + date[1:5]+ '/' + date[5:7] + '/B' + date[1:],'r')
                MSL+=B.variables['MSL'][0,la0:la1,lo0:lo1]
        
            avc = len(d[sea][re])
            cf = ax.contourf(LON[lons],LAT[lats],PV/avc,cmap=matplotlib.cm.BrBG,levels=levels,extend='both')
            cl = ax.contour(LON[lons],LAT[lats],MSL/100/avc,levels=np.arange(970,1040,5),colors='purple',linewidths=0.5,zorder=3)
            plt.clabel(cl, inline=1, fontsize=6, fmt='%d')
            cbax = fig.add_axes([0, 0, 0.1, 0.1])
            cbar=plt.colorbar(cf, ticks=levels,cax=cbax)
            func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.01)
            fig.canvas.mpl_connect('draw_event', func)
            cbar.ax.set_xlabel('PVU')
            plt.subplots_adjust(wspace=0,hspace=0)
        
            fig.savefig(pii + '%s-%s-avPV-at-%02d.png'%(sea,re,k),dpi=300,bbox_inches='tight')
            plt.close('all')

