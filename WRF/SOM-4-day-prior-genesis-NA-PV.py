import pandas as pd
import os
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import sys
from netCDF4 import Dataset as ds
import numpy as np
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert

d = pd.read_csv('/home/ascherrmann/scripts/WRF/SOM-genesis-cluster-IDs.csv',header=None)

minlon = -80
minlat = 30
maxlat = 80
maxlon = 10

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1


ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/4regionsPV/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/SOM/'

fig = plt.figure(figsize=(20,10))
gs = gridspec.GridSpec(ncols=6, nrows=4)
cmap = matplotlib.cm.BrBG
levels = np.arange(-3,3.5,0.5)

for k in range(4):
    for l in range(6):
        n = (k*6+l)
        row = d.iloc[n]
        ax = fig.add_subplot(gs[k,l],projection=ccrs.PlateCarree())
        if k==0 and l==5:
            axup = ax
            uppos = ax.get_position()
            yup = uppos.y0 + uppos.height
        if k==3 and l==5:
            axlow = ax
            lowpos = ax.get_position()
            ylow = lowpos.y0

#        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
        ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())
        avPV = np.zeros((lats.size,lons.size))

        counter=0
#        if k!=1 or l!=2:
#            continue
#        print('')
#        print(n)
        for r in row:
            if r==0:
                continue
            counter+=1
            dirs = '%06d'%r
            dates = np.array([])
            for fi in os.listdir(ps + dirs + '/300/'):
                if fi.startswith('D'):
                    dates = np.append(dates,fi)

            date = dates[8]
#            print(date[1:])
            pv = ds(ps + dirs + '/300/' + date)
            avPV += pv.variables['PV'][0,0,la0:la1,lo0:lo1]
#        print(counter)
        avPV/=counter
        cf = ax.contourf(LON[lons],LAT[lats],avPV,cmap=cmap,levels=levels)

#plt.subplots_adjust(wspace=0,hspace=0,right=0.9)
plt.subplots_adjust(wspace=0,hspace=0,right=0.9,top=0.685)

uppos = axup.get_position()
yup = uppos.y0 + uppos.height
lowpos = axlow.get_position()
ylow = lowpos.y0

cbax = fig.add_axes([0.9, ylow, 0.01, yup-ylow])
cbar=plt.colorbar(cf, ticks=levels,cax=cbax)
#plt.subplots_adjust(wspace=0,hspace=0,right=0.9,top=0.7)
fig.savefig(pi + 'SOM-genesis-4-days-prior-NA.png',dpi=300,bbox_inches='tight')
plt.close('all')


