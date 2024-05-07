from netCDF4 import Dataset as ds
import numpy as np
import pandas as pd
import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append('/home/ascherrmann/scripts/')
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert

import helper
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec
import matplotlib
import wrf

regio = ['west','central','east']
sea = ['DJF']

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/4regionsPV/'
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
trackp = '/atmosdyn/michaesp/mincl.era-5/tracks/'
df = pd.read_csv('/atmosdyn2/ascherrmann/011-all-ERA5/data/pandas-basic-data-all-deep-over-sea-12h.csv')
df = df.iloc[np.where(df['reg'].values=='MED')[0]]
era5 = '/atmosdyn2/era5/cdf/'
LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

lons,lats = np.where((LON>=-119.5)&(LON<=80))[0],np.where((LAT>=10.5)&(LAT<=80))[0]
lo0,lo1,la0,la1 = lons[0],lons[-1],lats[0],lats[-1]

dID = df['ID'].values
htminSLP = df['htSLPmin'].values
mdates = df['dates'].values
months = df['months'].values

f = open(ps + '100-region-season.txt','rb')
di = pickle.load(f)
f.close()

dates = dict()
for s in sea:
    dates[s] = dict()
    for r in regio:
        dates[s][r] = np.array([])
        for ids in di[s][r]:
            loc = np.where(dID==ids)[0][0]
            dates[s][r] = np.append(dates[s][r],helper.change_date_by_hours(helper.change_date_by_hours(mdates[loc],-1*htminSLP[loc]+6),-96))

f = open('/atmosdyn2/ascherrmann/013-WRF-sim/data/Minus-4-days-prior-genesis-in-West-east-Med-cyclones-ERA5.txt','wb')
pickle.dump(dates,f)
f.close()

f = open('/atmosdyn2/ascherrmann/013-WRF-sim/data/Minus-4-days-prior-genesis-in-West-east-Med-cyclones-ERA5.txt','rb')
dates = pickle.load(f)
f.close()

## index of 300 hPa pressure level in Z file
pl = 3

avU = dict()
cmap=matplotlib.cm.BrBG
ulvl=np.arange(-10,11,2)
for s in sea:
    avU[s] = dict()
    for r in regio:
        avU[s][r] = np.zeros((361,720))
        for d in dates[s][r]:
            y = d[:4]
            m = d[4:6]
            Z = ds(era5 + '%s/%s/Z%s'%(y,m,d))
            U300 = Z.variables['U'][0,pl]
            Z.close()
            avU[s][r] += U300
        avU[s][r]/=dates[s][r].size
    
        fig=plt.figure(figsize=(8,6))
        gs = gridspec.GridSpec(nrows=1, ncols=1)
        ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=5, edgecolor='black')

        clim = ds('/atmosdyn2/ascherrmann/013-WRF-sim/' + s + '-clim/wrfout_d01_2000-12-01_00:00:00')
        pressure = wrf.getvar(clim,'pressure')
        U = wrf.getvar(clim,'U')
        U = (U[:,:,1:]+U[:,:,:-1])/2
        climU = wrf.interplevel(U,pressure,300,meta=False)

        print(avU[s][r][la0:la1+1,lo0:lo1+1].shape,climU.shape)
        h = ax.contourf(LON[lons],LAT[lats],avU[s][r][la0:la1+1,lo0:lo1+1]-climU,levels=ulvl,cmap=cmap)

        cbax = fig.add_axes([0, 0, 0.1, 0.1])
        cbar=plt.colorbar(h, ticks=ulvl,cax=cbax)
        func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.01)
        fig.canvas.mpl_connect('draw_event', func)
        fig.savefig(dwrf + 'image-output/%s-%s-minus-4-day-U-vs-climatology.png'%(s,r),dpi=300,bbox_inches='tight')
        plt.close('all')

