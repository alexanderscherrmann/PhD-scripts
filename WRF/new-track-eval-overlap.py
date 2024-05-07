from netCDF4 import Dataset as ds
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec
import matplotlib
minlon = -80
minlat = 30
maxlat = 85
maxlon = 60

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
era5 = '/atmosdyn2/era5/cdf/'

f = open('/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/general-track-types-size-0.2-streamer-types-0.8.txt','rb')
d = pickle.load(f)
f.close()
trange = np.arange(-168,49,3)

for k in d.keys():
    if len(d[k])>=60:
        
        fig = plt.figure(figsize=(9,4))
        gs = gridspec.GridSpec(ncols=2, nrows=1)
        ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
        ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())
        counters = np.zeros((len(lats),len(lons)))
        counterr = np.zeros_like(counters)
        PV = np.zeros_like(counters)
        MSL = np.zeros_like(counters)
        times = np.array([])
        le = len(d[k])
        
        for t in d[k]:
            dates = np.array([])
            dirs,kk = t[:6],int(t[7:9])
            sv,rv = eval(t[10:])
            sm = ds(ps + dirs + '/300/streamer-mask.nc','r')
            rm = ds(ps + dirs + '/300/ridge-mask.nc','r')
            sm = sm.variables['mask'][kk,la0:la1,lo0:lo1]
            rm = rm.variables['mask'][kk,la0:la1,lo0:lo1]
            for f in os.listdir(ps + dirs + '/300/'):
                if f.startswith('D'):
                    dates = np.append(dates,f)
            date = dates[kk]
            S=ds(ps + dirs + '/300/' + date,'r')
            PV+=S.variables['PV'][0,0,la0:la1,lo0:lo1]
            #PVr[rm==rv]+=S.variables['PV'][0,0,la0:la1,lo0:lo1][rm==rv]
            B = ds(era5 + date[1:5]+ '/' + date[5:7] + '/B' + date[1:],'r')
            MSL+=B.variables['MSL'][0,la0:la1,lo0:lo1]
            counters[sm==sv]+=1
            counterr[rm==rv]+=1

            times = np.append(times,trange[kk])

        ax.contour(LON[lons],LAT[lats],counters*100/le,levels=np.arange(20,101,10),colors=['saddlebrown','grey','dodgerblue','blue','cyan','purple','orchid','orange','red'],linewidths=1,linestyles='-')
        ax.contour(LON[lons],LAT[lats],counterr*100/le,levels=np.arange(20,101,10),colors=['saddlebrown','grey','dodgerblue','blue','cyan','purple','orchid','orange','red'],linewidths=1,linestyles='--')

        ax.text(0.05,0.925,'%d'%le,transform=ax.transAxes,fontsize=6,fontweight='bold')
        
        ax=fig.add_subplot(gs[0,1],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
        ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

        counter = counters+counterr
        ax.contourf(LON[lons],LAT[lats],PV/counter,cmap=matplotlib.cm.BrBG,levels=np.arange(-5,5.5,0.5))
        ax.contour(LON[lons],LAT[lats],MSL/le,levels=np.arange(970,1040,5),colors='purple')
        fig.savefig(pi + 'general-overlap-%s-%d'%(dirs,kk),dpi=300,bbox_inches='tight')
        plt.close('all')


        fig = plt.figure(figsize=(6,4))
        gs = gridspec.GridSpec(ncols=1, nrows=1)
        ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
        ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

        

        fig,ax =plt.subplots()
        t,co = np.unique(times,return_counts=True)
        ax.bar(t,co)
        fig.savefig(pi + 'general-overlap-%s-%d-counts'%(dirs,kk),dpi=300,bbox_inches='tight')
        plt.close('all')
