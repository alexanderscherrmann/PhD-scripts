import numpy as np
from wrf import interplevel as intp
from netCDF4 import Dataset as ds
from numpy.random import randint as ran
import pandas as pd
import pickle
import wrf
from wrf import interplevel as intp

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import os
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
PD = '/atmosdyn2/ascherrmann/009-ERA-5/MED/data/'

dfD = pd.read_csv(ps + 'DJF-intense-cyclones.csv')
dfS = pd.read_csv(ps + 'SON-intense-cyclones.csv')

f = open(ps + 'DJF-boot-strap-U-data.txt','rb')
avDJFU = pickle.load(f)
f.close()

f = open(ps + 'DJF-boot-strap-T-data.txt','rb')
avDJFT = pickle.load(f)
f.close()

### plotting region
minlon = -120
minlat = 10
maxlat = 80
maxlon = 80

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)
lonticks=np.arange(minlon, maxlon+1,20)
latticks=np.arange(minlat, maxlat+1,10)
lonshort = np.linspace(-120,80,401)
latshort = np.linspace(10,80,141)

### dictionaries to save

DJFclim = ds('/atmosdyn2/ascherrmann/013-WRF-sim/DJF-clim/wrfout_d01_2000-12-01_00:00:00','r')
Pwrf = wrf.getvar(DJFclim,'pressure',meta=False)
lonwrf = np.linspace(-120,79.5,400)
latwrf = np.linspace(10,79.5,140)
Uwrf = wrf.getvar(DJFclim,'U',meta=False)
Uwrf = Uwrf[:,:,1:]/2 + Uwrf[:,:,:-1]/2
Uwrf_300hPa = intp(Uwrf,Pwrf,300.0,meta=False)
THwrf = wrf.getvar(DJFclim,'T',meta=False)
THwrf_850hPa = intp(THwrf,Pwrf,850,meta=False) + 300

anU = dict()
anT = dict()

pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/season/DJF/bootstrap/'
if not os.path.isdir(pi):
    os.mkdir(pi)

f = open(ps + 'DJF-individual-fields.txt','rb')
data = pickle.load(f)
f.close()

when = ['fourdaypriormature','fivedaypriormature','sixdaypriormature','sevendaypriormature','threedaypriormature','twodaypriormature','onedaypriormature','dates']

wl = ['4','5','6','7','3','2','1','0']

for l in avDJFU.keys():
    anU[l] = np.zeros_like(avDJFU[l])
    anT[l] = np.zeros_like(avDJFT[l])
    for k in range(1000):
        anU[l][k,:-1,:-1] = avDJFU[l][k,:-1,:-1]-Uwrf_300hPa
        anT[l][k,:-1,:-1] = avDJFT[l][k,:-1,:-1]-THwrf_850hPa


for wq,we in zip(wl,when):
  if wq=='12':
    sd4 = data['DJF']['intense-cyclones.csv'][200][we]
    
    plotU = dict()
    plotT = dict()
    for l in avDJFU.keys():
        ids = dfD['ID'].values[np.where(dfD['region'].values==l)[0]]
        c = len(ids)
    
        plotU[l] = np.zeros(avDJFT[l][0].shape)
        plotT[l] = np.zeros(avDJFT[l][0].shape)
    
        clusavU = np.zeros(avDJFT[l][0].shape)
        clusavT = np.zeros(avDJFT[l][0].shape)
        
        clusstdU = np.zeros((c,avDJFT[l][0].shape[0],avDJFT[l][0].shape[1]))
        clusstdT = np.zeros((c,avDJFT[l][0].shape[0],avDJFT[l][0].shape[1]))

        for q,i in enumerate(ids):
            clusavU[:-1,:-1] +=(sd4[i]['U300hPa'][:-1,:-1]-Uwrf_300hPa)/c
            clusavT[:-1,:-1] +=(sd4[i]['TH850'][:-1,:-1]-THwrf_850hPa)/c
            clusstdU[q,:-1,:-1] +=(sd4[i]['U300hPa'][:-1,:-1]-Uwrf_300hPa)/c
            clusstdT[q,:-1,:-1] +=(sd4[i]['TH850'][:-1,:-1]-THwrf_850hPa)/c


        for j in range(avDJFU[l].shape[1]-1):
            for i in range(avDJFU[l].shape[2]-1):
                if (clusavU[j,i]/np.std(clusstdU[:,j,i])>np.percentile(np.sort(anU[l][:,j,i]),95)/np.std(anU[l][:,j,i])) or (clusavU[j,i]<0 and clusavU[j,i]/np.std(clusstdU[:,j,i])<np.percentile(anU[l][:,j,i],5)/np.std(anU[l][:,j,i])):
                #if (clusavU[j,i]>np.percentile(np.sort(anU[l][:,j,i]),95)) or (clusavU[j,i]<0 and clusavU[j,i]<np.percentile(anU[l][:,j,i],5)):
                    plotU[l][j,i] = clusavU[j,i]
                
                else:
                    plotU[l][j,i] =np.NaN

                if (clusavT[j,i]/np.std(clusstdT[:,j,i])>np.percentile(np.sort(anT[l][:,j,i]),95)/np.std(anT[l][:,j,i])) or (clusavT[j,i]<0 and clusavT[j,i]/np.std(clusstdT[:,j,i])<np.percentile(anT[l][:,j,i],5)/np.std(anT[l][:,j,i])):
                #if (clusavT[j,i]>np.percentile(np.sort(anT[l][:,j,i]),95)) or (clusavT[j,i]<0 and clusavT[j,i]<np.percentile(anT[l][:,j,i],5)): 
                    plotT[l][j,i] = clusavT[j,i]
    
                else:
                    plotT[l][j,i] =np.NaN
            
    
    fig = plt.figure(figsize=(8,20))
    gs = gridspec.GridSpec(nrows=2, ncols=3)
    axes = []
    
    for k in range(2):
        for l in range(3):
            axes.append(fig.add_subplot(gs[k,l],projection=ccrs.PlateCarree()))
    
    cmap = matplotlib.cm.BrBG
    levels = np.arange(-15,16,3)
    norm = BoundaryNorm(levels,cmap.N)
    unit = 'm s$^{-1}$'
    lvl=levels
    
    for ax,l in zip(axes,avDJFU.keys()):
    
        ids = dfD['ID'].values[np.where(dfD['region'].values==l)[0]]
        c = len(ids)
        
        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
        ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())
        h = ax.contourf(lonwrf,latwrf,plotU[l][:-1,:-1],cmap=cmap,norm=norm,levels=levels,extend='both')
        ax.set_xticks([])
        ax.set_yticks([])
    #    ax.set_xticks(lonticks, crs=ccrs.PlateCarree())
    #    ax.set_yticks(latticks, crs=ccrs.PlateCarree())
    #    ax.set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
    #    ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)
    
        ax.text(0.025,1.02,l + '-%d'%c,fontsize=8,fontweight='bold',color='k',transform=ax.transAxes)
    
    plt.subplots_adjust(right=0.9,wspace=0.0,hspace=0.0,top=0.2,bottom=0.1)
    cbax = fig.add_axes([0.9, 0.107, 0.01, 0.0865])
    cbar=plt.colorbar(h, ticks=lvl,cax=cbax)
    
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_xlabel(unit,fontsize=10)
    cbar.ax.set_xticklabels(lvl)
    
    name = 'U-boot-anomaly-cluster-' + wq + '.png'
    fig.savefig(pi + name,dpi=300,bbox_inches='tight')
    plt.close('all')
    
    
    fig = plt.figure(figsize=(8,20))
    gs = gridspec.GridSpec(nrows=2, ncols=3)
    axes = []
    for k in range(2):
        for l in range(3):
            axes.append(fig.add_subplot(gs[k,l],projection=ccrs.PlateCarree()))
    
    cmap = matplotlib.cm.RdBu.reversed()
    levels = np.arange(-5,6,1)
    norm = BoundaryNorm(levels,cmap.N)
    unit = 'K'
    lvl=levels
    
    for ax,l in zip(axes,avDJFU.keys()):
    
        ids = dfD['ID'].values[np.where(dfD['region'].values==l)[0]]
        c = len(ids)
    
        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
        ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())
        h = ax.contourf(lonwrf,latwrf,plotT[l][:-1,:-1],cmap=cmap,norm=norm,levels=levels,extend='both')
        ax.set_xticks([])
        ax.set_yticks([])
    #    ax.set_xticks(lonticks, crs=ccrs.PlateCarree())
    #    ax.set_yticks(latticks, crs=ccrs.PlateCarree())
    #    ax.set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
    #    ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)
    
        ax.text(0.025,1.02,l + '-%d'%c,fontsize=8,fontweight='bold',color='k',transform=ax.transAxes)
    
    plt.subplots_adjust(right=0.9,wspace=0.0,hspace=0.0,top=0.2,bottom=0.1)
    cbax = fig.add_axes([0.9, 0.107, 0.01, 0.0865])
    cbar=plt.colorbar(h, ticks=lvl,cax=cbax)
    
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_xlabel(unit,fontsize=10)
    cbar.ax.set_xticklabels(lvl)
    
    name = 'T-boot-anomaly-cluster-' + wq  + '.png'
    fig.savefig(pi + name,dpi=300,bbox_inches='tight')
    plt.close('all')

f = open(ps + 'SON-boot-strap-U-data.txt','rb')
avSONU = pickle.load(f)
f.close()

f = open(ps + 'SON-boot-strap-T-data.txt','rb')
avSONT = pickle.load(f)
f.close()

SONclim = ds('/atmosdyn2/ascherrmann/013-WRF-sim/SON-clim/wrfout_d01_2000-12-01_00:00:00','r')
Pwrf = wrf.getvar(SONclim,'pressure',meta=False)
lonwrf = np.linspace(-120,79.5,400)
latwrf = np.linspace(10,79.5,140)
Uwrf = wrf.getvar(SONclim,'U',meta=False)
Uwrf = Uwrf[:,:,1:]/2 + Uwrf[:,:,:-1]/2
Uwrf_300hPa = intp(Uwrf,Pwrf,300.0,meta=False)
THwrf = wrf.getvar(SONclim,'T',meta=False)
THwrf_850hPa = intp(THwrf,Pwrf,850,meta=False) + 300

anU = dict()
anT = dict()

pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/season/SON/bootstrap/'
if not os.path.isdir(pi):
    os.mkdir(pi)

f = open(ps + 'SON-individual-fields.txt','rb')
data = pickle.load(f)
f.close()

for l in avSONU.keys():
    anU[l] = np.zeros_like(avSONU[l])
    anT[l] = np.zeros_like(avSONT[l])
    for k in range(1000):
        anU[l][k,:-1,:-1] = avSONU[l][k,:-1,:-1]-Uwrf_300hPa
        anT[l][k,:-1,:-1] = avSONT[l][k,:-1,:-1]-THwrf_850hPa


for wq,we in zip(wl,when):
    sd4 = data['SON']['intense-cyclones.csv'][200][we]

    plotU = dict()
    plotT = dict()
    for l in avSONU.keys():
        ids = dfS['ID'].values[np.where(dfS['region'].values==l)[0]]
        c = len(ids)

        plotU[l] = np.zeros(avSONT[l][0].shape)
        plotT[l] = np.zeros(avSONT[l][0].shape)

        clusavU = np.zeros(avSONT[l][0].shape)
        clusavT = np.zeros(avSONT[l][0].shape)

        for i in ids:
            clusavU[:-1,:-1] +=(sd4[i]['U300hPa'][:-1,:-1]-Uwrf_300hPa)/c
            clusavT[:-1,:-1] +=(sd4[i]['TH850'][:-1,:-1]-THwrf_850hPa)/c

        for j in range(avSONU[l].shape[1]-1):
            for i in range(avSONU[l].shape[2]-1):
                if (clusavU[j,i]>np.percentile(np.sort(anU[l][:,j,i]),95)) or (clusavU[j,i]<0 and clusavU[j,i]<np.percentile(anU[l][:,j,i],5)):
                    plotU[l][j,i] = clusavU[j,i]

                else:
                    plotU[l][j,i] =np.NaN


                if (clusavT[j,i]>np.percentile(np.sort(anT[l][:,j,i]),95)) or (clusavT[j,i]<0 and clusavT[j,i]<np.percentile(anT[l][:,j,i],5)):
                    plotT[l][j,i] = clusavT[j,i]

                else:
                    plotT[l][j,i] =np.NaN


    fig = plt.figure(figsize=(8,20))
    gs = gridspec.GridSpec(nrows=2, ncols=3)
    axes = []

    for k in range(2):
        for l in range(3):
            axes.append(fig.add_subplot(gs[k,l],projection=ccrs.PlateCarree()))

    cmap = matplotlib.cm.BrBG
    levels = np.arange(-15,16,3)
    norm = BoundaryNorm(levels,cmap.N)
    unit = 'm s$^{-1}$'
    lvl=levels

    for ax,l in zip(axes,avSONU.keys()):

        ids = dfS['ID'].values[np.where(dfS['region'].values==l)[0]]
        c = len(ids)

        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
        ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())
        h = ax.contourf(lonwrf,latwrf,plotU[l][:-1,:-1],cmap=cmap,norm=norm,levels=levels,extend='both')
        ax.set_xticks([])
        ax.set_yticks([])

        ax.text(0.025,1.02,l + '-%d'%c,fontsize=8,fontweight='bold',color='k',transform=ax.transAxes)

    plt.subplots_adjust(right=0.9,wspace=0.0,hspace=0.0,top=0.2,bottom=0.1)
    cbax = fig.add_axes([0.9, 0.107, 0.01, 0.0865])
    cbar=plt.colorbar(h, ticks=lvl,cax=cbax)

    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_xlabel(unit,fontsize=10)
    cbar.ax.set_xticklabels(lvl)

    name = 'U-boot-anomaly-cluster-' + wq + '.png'
    fig.savefig(pi + name,dpi=300,bbox_inches='tight')
    plt.close('all')


    fig = plt.figure(figsize=(8,20))
    gs = gridspec.GridSpec(nrows=2, ncols=3)
    axes = []
    for k in range(2):
        for l in range(3):
            axes.append(fig.add_subplot(gs[k,l],projection=ccrs.PlateCarree()))

    cmap = matplotlib.cm.RdBu.reversed()
    levels = np.arange(-5,6,1)
    norm = BoundaryNorm(levels,cmap.N)
    unit = 'K'
    lvl=levels

    for ax,l in zip(axes,avSONU.keys()):

        ids = dfS['ID'].values[np.where(dfS['region'].values==l)[0]]
        c = len(ids)

        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
        ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

        h = ax.contourf(lonwrf,latwrf,plotT[l][:-1,:-1],cmap=cmap,norm=norm,levels=levels,extend='both')

        ax.set_xticks([])
        ax.set_yticks([])
    #    ax.set_xticks(lonticks, crs=ccrs.PlateCarree())
    #    ax.set_yticks(latticks, crs=ccrs.PlateCarree())
    #    ax.set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
    #    ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)

        ax.text(0.025,1.02,l + '-%d'%c,fontsize=8,fontweight='bold',color='k',transform=ax.transAxes)

    plt.subplots_adjust(right=0.9,wspace=0.0,hspace=0.0,top=0.2,bottom=0.1)
    cbax = fig.add_axes([0.9, 0.107, 0.01, 0.0865])
    cbar=plt.colorbar(h, ticks=lvl,cax=cbax)

    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_xlabel(unit,fontsize=10)
    cbar.ax.set_xticklabels(lvl)

    name = 'T-boot-anomaly-cluster-' + wq  + '.png'
    fig.savefig(pi + name,dpi=300,bbox_inches='tight')
    plt.close('all')
