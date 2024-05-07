import numpy as np
import os
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')

import xarray as xr
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import helper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patch
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle
import os

def colbar(cmap,minval,maxval,nlevels):
    maplist = [cmap(i) for i in range(cmap.N)]
    newmap = ListedColormap(maplist)
    norm = BoundaryNorm(pvr_levels,cmap.N)
    return newmap, norm

LON = np.round(np.linspace(-180,180,721),1)
LAT = np.round(np.linspace(0,90,361),1)

def find_nearest_grid_point(lon,lat):

    dlon = LON-lon
    dlat = LAT-lat

    lonid = np.where(abs(dlon)==np.min(abs(dlon)))[0][0]
    latid = np.where(abs(dlat)==np.min(abs(dlat)))[0][0]

    return lonid,latid


pload = '/atmosdyn2/ascherrmann/009-ERA-5/MED/ctraj/use/'
psave = '/atmosdyn2/ascherrmann/paper/cyc-env-PV/'

f = open(pload+'PV-data-dPSP-100-ZB-800-2-400-correct-distance.txt','rb')
data = pickle.load(f)
f.close()

datadi = data['rawdata']
dit = data['dit']
gridmap = dict()
pvrs = ['DELTAPV']

NORO = xr.open_dataset('/home/ascherrmann/scripts/ERA5-utils/NORO')
ZB = NORO['ZB'].values[0]
Zlon = NORO['lon']
Zlat = NORO['lat']
Eminv = 800
#elevation_levels = np.arange(Eminv,Emaxv,400)
elevation_levels = np.array([Eminv,100000])#1600,2400])

ecounter = np.zeros((LAT.size,LON.size))
nccounter = np.zeros((LAT.size,LON.size))
ccounter = np.zeros((LAT.size,LON.size))
necounter = np.zeros((LAT.size,LON.size))

gecounter = np.zeros((LAT.size,LON.size))
gccounter = np.zeros((LAT.size,LON.size))

PVedge=0.75

cyc = 'cyc'
env = 'env'
if not os.path.isfile('frequency-PV-data-cyc-env-0.15.txt'):
    abscounter = 0
    for q, date in enumerate(datadi.keys()):
        abscounter+=datadi[date]['PV'].shape[0] * (datadi[date]['PV'].shape[1]-1)
        idp = np.where(datadi[date]['PV'][:,0]>=PVedge)[0]
        tralon = datadi[date]['lon'][idp]
        tralat = datadi[date]['lat'][idp]
        datadi[date]['DELTAPV'] = np.zeros(datadi[date]['PV'].shape)
        datadi[date]['DELTAPV'][:,1:] = datadi[date]['PV'][:,:-1]-datadi[date]['PV'][:,1:]
    
        for k in range(len(idp)):
            for l in range(len(tralon[0])):
                lon = tralon[k,l]
                lat = tralat[k,l]
                lonid,latid = find_nearest_grid_point(lon,lat)
                
                if dit[date][cyc][idp[k],l] ==1:
                    gccounter[latid,lonid]+=1
                else:
                    gecounter[latid,lonid]+=1
    
    
                if datadi[date]['DELTAPV'][idp[k],l] >=0.15:
                  if dit[date][cyc][idp[k],l] ==1:
    #                gridmap['DELTAPV'][latid,lonid]+=datadi[date]['DELTAPV'][idp[k],l]
                    ccounter[latid,lonid]+=1
                  else:
    #                gridmap['DELTAPV'][latid,lonid]+=datadi[date]['DELTAPV'][idp[k],l]
                    ecounter[latid,lonid]+=1
    
                if datadi[date]['DELTAPV'][idp[k],l] <=-0.15:
                  if dit[date][cyc][idp[k],l] ==1:
    #                gridmap['nPV'][latid,lonid]+=datadi[date]['DELTAPV'][idp[k],l]
                    nccounter[latid,lonid]+=1
                  else:
    #                gridmap['DELTAPV'][latid,lonid]+=datadi[date]['DELTAPV'][idp[k],l]
                    necounter[latid,lonid]+=1
    
    gridmap['nccounter'] = nccounter
    gridmap['ccounter'] = ccounter
    gridmap['necounter'] = necounter
    gridmap['ecounter'] = ecounter
    
    gridmap['gccounter'] = gccounter
    gridmap['gecounter'] = gecounter
    
    f = open(psave + 'frequency-PV-data-cyc-env-0.15.txt','wb')
    pickle.dump(gridmap,f)
    f.close()

f = open(psave + 'frequency-PV-data-cyc-env-0.15.txt','rb')
gridmap = pickle.load(f)
f.close()

ccounter = gridmap['ccounter']
nccounter = gridmap['nccounter']
ecounter = gridmap['ecounter']
necounter = gridmap['necounter']
gccounter = gridmap['gccounter']
gecounter = gridmap['gecounter']

what = ['ccounter','nccounter','ecounter','necounter','gccounter','gecounter']
sums = np.zeros(len(what))
abscounter = np.sum(gccounter + gecounter)

for q,k in enumerate(what):
    loc = np.where(gridmap[k]!=0)
    sums[q] = np.sum(gridmap[k][loc])
#    gridmap[k][loc]=gridmap[k][loc]/np.sum(gridmap[k][loc])
#    gridmap[k]/=abscounter   
    loc = np.where(gridmap[k]==0)
    gridmap[k][loc]=np.nan


alpha=1.
linewidth=.2

minpltlatc = 25
minpltlonc = -20

maxpltlatc = 60
maxpltlonc = 50


#for pv in gridmap.keys():
#    #np.append('counter','ncounter'):
#    minv=np.nanmin(gridmap[pv])*100
#    maxv=np.nanmax(gridmap[pv])*100
#    print(minv,maxv,(maxv-minv)/10)
#    
#    steps=(maxv-minv)/10
#    pvr_levels = np.arange(minv,maxv+0.0001,steps)
#    gridmap[pv][np.where(gridmap[pv]*100<=pvr_levels[1])]=np.nan
#    ap = plt.cm.nipy_spectral
#    cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
#    ticklabels=pvr_levels
#
#    fig = plt.figure(figsize=(6,4))
#    gs = gridspec.GridSpec(nrows=1, ncols=1)
#    ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
#    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7) 
#    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=2, edgecolor='black')
#    ax.contour(Zlon,Zlat,ZB,levels = elevation_levels,colors='k',linewidths=0.35,alpha=1)
#
#    lc=ax.contourf(LON,LAT,gridmap[pv]*100,levels=pvr_levels,cmap=cmap,norm=norm,zorder=1)
#    
#    lonticks=np.arange(minpltlonc, maxpltlonc,5)
#    latticks=np.arange(minpltlatc, maxpltlatc,5)
#    
#    ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
#    ax.set_yticks(latticks, crs=ccrs.PlateCarree());
##    ax.set_xticklabels(labels=lonticks,fontsize=8)
##    ax.set_yticklabels(labels=latticks,fontsize=8)
#    
##    ax.xaxis.set_major_formatter(LongitudeFormatter())
##    ax.yaxis.set_major_formatter(LatitudeFormatter())
#    
#    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
#    
#    cbax = fig.add_axes([0, 0, 0.1, 0.1])
#    cbar=plt.colorbar(lc, ticks=pvr_levels,cax=cbax)
#    
#    func=resize_colorbar_vert(cbax, ax, pad=0, size=0.02)
#    fig.canvas.mpl_connect('draw_event', func)
#    
##    cbar.ax.tick_params(labelsize=8)
#    cbar.ax.set_yticklabels(np.append(r'%',np.round(ticklabels[1:],3)))
#    cbar.ax.set_xlabel('\%')
#    figname = psave + pv +'-gridmap' + '-PVedge-' + str(PVedge) + '.png'
#    
#    fig.savefig(figname,dpi=300,bbox_inches="tight")
#    plt.close()


fig = plt.figure(figsize=(10,4))
gs = gridspec.GridSpec(nrows=1, ncols=2)
div = np.sum(sums)

maxval = np.zeros(4)
for k in range(2):
    for l in range(2):
#        maxval[k*2+l] = np.nanmax(gridmap[what[k*2 + l]]*sums[k*2+l]/div*100)
        maxval[k*2+l] = np.nanmax(gridmap[what[k*2 + l]]*100)

minv,maxv = 0,np.max(maxval)*0.4
steps = maxv/6
pvr_levels = np.arange(minv,maxv+0.0001,steps)
ap = plt.cm.YlGnBu
cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
ticklabels=pvr_levels

l = 1
lab=['a)','b)']
for k in range(2):
#    for l in range(2):
#        gridmap[what[k*2 + l]][np.where(gridmap[what[k*2 + l]]*sums[k*2+l]/div*100<=pvr_levels[1])]=np.nan
#        ax=fig.add_subplot(gs[k,l],projection=ccrs.PlateCarree())
        ax=fig.add_subplot(gs[0,k],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=2, edgecolor='black')
        ax.contour(Zlon,Zlat,ZB,levels=elevation_levels,colors='purple',linewidths=0.35)
        lc=ax.contourf(LON,LAT,gridmap[what[k*2 + l]]*100,levels=pvr_levels,cmap=cmap,norm=norm,zorder=1,extend='max')
        ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
        ax.text(0.025,0.9,lab[k],fontsize=10,fontweight='bold',transform=ax.transAxes)

#cbax = fig.add_axes([0.7, 0.11, 0.01, 0.58])
#cax = plt.axes([0.8,0.1,0.02,0.675])
cax=plt.axes([0.775,0.1,0.01,0.4])
plt.subplots_adjust(wspace=0,hspace=0,right=0.775,bottom=0.1,top=0.5)
#plt.subplots_adjust(wspace=0,hspace=0,bottom=0.1,top=0.7,right=0.7)
cbar=plt.colorbar(lc, ticks=pvr_levels,cax=cax)
#func=resize_colorbar_vert(cbax, ax, pad=0, size=0.02)
cbar.ax.set_yticklabels(np.append(r'%',np.round(ticklabels[1:],4)))
figname = psave +'negative-all-gridmap' + '-PVedge-' + str(PVedge) + '.png'
#fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close('all')


###
### absolute trajectory numbers
###

for k in range(2):
    for l in range(2):
        maxval[k*2+l] = np.nanmax(gridmap[what[k*2 + l]])
        print(np.nanmax(gridmap[what[k*2 + l]]))

#minv,maxv = 0,np.max(maxval)*0.4
#steps = maxv/6
pvr_levels = np.array([0,200,400,600,800,1100,1500,2000,2500,3000])
cmap = plt.cm.YlGnBu
norm=BoundaryNorm(pvr_levels,cmap.N)
#pvr_levels = np.arange(minv,maxv+0.0001,steps).astype(int)

#ap = plt.cm.YlGnBu
#cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
#ticklabels=np.round(pvr_levels,1)
ticklabels=pvr_levels
#print('maxval',maxv)

fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(nrows=2, ncols=2)
lab=['(a)','(b)','(c)','(d)']
for k in range(2):
    for l in range(2):
        ax=fig.add_subplot(gs[l,k],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=2, edgecolor='black')
        ax.contour(Zlon,Zlat,ZB,levels=elevation_levels,colors='red',linewidths=0.35,zorder=10)
        lc=ax.contourf(LON,LAT,gridmap[what[k*2+l]],levels=pvr_levels,cmap=cmap,norm=norm,zorder=1,extend='max')
        ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
        ax.text(0.025,0.9,lab[k+l*2],fontsize=14,transform=ax.transAxes)

cax=plt.axes([0.775,0.1,0.01,0.405])
plt.subplots_adjust(wspace=0,hspace=0,right=0.775,bottom=0.1,top=0.505)
cbar=plt.colorbar(lc, ticks=pvr_levels,cax=cax)
cbar.ax.set_yticklabels(ticklabels)
figname = psave +'absolute-trajectory-gridmap-PVedge-' + str(PVedge) + '.png'
fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close('all')



###
###  negative relative map
###


fig = plt.figure(figsize=(10,4))
gs = gridspec.GridSpec(nrows=1, ncols=2)
gridmap['nccounter']/=gridmap['gccounter']
gridmap['necounter']/=gridmap['gecounter']
#loc = np.where(gridmap['gccounter']*abscounter<100)
#gridmap['nccounter'][loc]=np.nan

for k in range(2):
    for l in range(2):
        maxval[k*2+l] = np.nanmax(gridmap[what[k*2 + l]]*100)

minv,maxv = 0,np.max(maxval)*0.5
steps = maxv/6
pvr_levels = np.arange(minv,maxv+0.0001,steps)
ap = plt.cm.YlGnBu
cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
ticklabels=pvr_levels

l = 1
lab=['a)','b)']
for k in range(2):
    ax=fig.add_subplot(gs[0,k],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=2, edgecolor='black')
    ax.contour(Zlon,Zlat,ZB,levels=elevation_levels,colors='purple',linewidths=0.35)
    lc=ax.contourf(LON,LAT,gridmap[what[k*2+l]]*100,levels=pvr_levels,cmap=cmap,norm=norm,zorder=1,extend='max')
    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
    ax.text(0.025,0.9,lab[k],fontsize=10,fontweight='bold',transform=ax.transAxes)


cax=plt.axes([0.775,0.1,0.01,0.4])
plt.subplots_adjust(wspace=0,hspace=0,right=0.775,bottom=0.1,top=0.5)
#plt.subplots_adjust(wspace=0,hspace=0,bottom=0.1,top=0.7,right=0.7)
cbar=plt.colorbar(lc, ticks=pvr_levels,cax=cax)
#func=resize_colorbar_vert(cbax, ax, pad=0, size=0.02)
cbar.ax.set_yticklabels(np.append(r'%',np.round(ticklabels[1:],4)))
figname = psave +'negative-realative-gridmap' + '-PVedge-' + str(PVedge) + '.png'
#fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close('all')


###
### positive relative map
###

fig = plt.figure(figsize=(10,4))
gs = gridspec.GridSpec(nrows=1, ncols=2)
gridmap['ccounter']/=gridmap['gccounter']
gridmap['ecounter']/=gridmap['gecounter']
print(np.nanpercentile(gridmap['gccounter']*abscounter,10),np.nanpercentile(gridmap['gccounter']*abscounter,25),np.nanpercentile(gridmap['gccounter']*abscounter,50),np.nanpercentile(gridmap['gccounter']*abscounter,75),np.nanpercentile(gridmap['gccounter']*abscounter,90),np.nanmax(gridmap['gccounter']*abscounter))

#loc = np.where(gridmap['gccounter']*abscounter<100)
#gridmap['ccounter'][loc]=np.nan

for k in range(2):
    for l in range(2):
        maxval[k*2+l] = np.nanmax(gridmap[what[k*2 + l]]*100)

minv,maxv = 0,np.max(maxval)*0.5
steps = maxv/6
pvr_levels = np.arange(minv,maxv+0.0001,steps)
ap = plt.cm.YlGnBu
cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
ticklabels=np.round(pvr_levels,1)

l = 1
lab=['a)','b)']
for k in range(2):
    ax=fig.add_subplot(gs[0,k],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=2, edgecolor='black')
    ax.contour(Zlon,Zlat,ZB,levels=elevation_levels,colors='purple',linewidths=0.35)
    lc=ax.contourf(LON,LAT,gridmap[what[k*2]]*100,levels=pvr_levels,cmap=cmap,norm=norm,zorder=1,extend='max')
    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
    ax.text(0.025,0.9,lab[k],fontsize=10,fontweight='bold',transform=ax.transAxes)


cax=plt.axes([0.775,0.1,0.01,0.4])
plt.subplots_adjust(wspace=0,hspace=0,right=0.775,bottom=0.1,top=0.5)
#plt.subplots_adjust(wspace=0,hspace=0,bottom=0.1,top=0.7,right=0.7)
cbar=plt.colorbar(lc, ticks=pvr_levels,cax=cax)
#func=resize_colorbar_vert(cbax, ax, pad=0, size=0.02)
cbar.ax.set_yticklabels(np.append(r'%',np.round(ticklabels[1:],4)))
figname = psave +'positive-realative-gridmap' + '-PVedge-' + str(PVedge) + '.png'
#fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close('all')


###
### both combined negative and positive relative frequency
### 


fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(nrows=2, ncols=2)
lab=['a)','b)','c)','d)']
for k in range(2):
    for l in range(2):
        ax=fig.add_subplot(gs[l,k],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=2, edgecolor='black')
        ax.contour(Zlon,Zlat,ZB,levels=elevation_levels,colors='purple',linewidths=0.35)
        lc=ax.contourf(LON,LAT,gridmap[what[k*2+l]]*100,levels=pvr_levels,cmap=cmap,norm=norm,zorder=1,extend='max')
        ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
        ax.text(0.025,0.9,lab[k+l*2],fontsize=10,fontweight='bold',transform=ax.transAxes)

cax=plt.axes([0.775,0.1,0.01,0.4])
plt.subplots_adjust(wspace=0,hspace=0,right=0.775,bottom=0.1,top=0.5)
cbar=plt.colorbar(lc, ticks=pvr_levels,cax=cax)
cbar.ax.set_yticklabels(np.append(r'%',np.round(ticklabels[1:],4)))
figname = psave +'relative-gridmap' + '-PVedge-' + str(PVedge) + '.png'
#fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close('all')

###
### differnece between positive and negative frequency
###

fig = plt.figure(figsize=(10,4))
gs = gridspec.GridSpec(nrows=1, ncols=2)
ap=matplotlib.cm.PiYG
pvr_levels=np.arange(-30,35,5)
minv,maxv=np.min(pvr_levels),np.max(pvr_levels)
cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
ticklabels=pvr_levels
what=['divc','dive']
gridmap['divc'] = gridmap['ccounter']-gridmap['nccounter']
gridmap['dive'] = gridmap['ecounter']-gridmap['necounter']
gridmap['divc'][abs(gridmap['divc'])<=5/100] = np.nan
gridmap['dive'][abs(gridmap['dive'])<=5/100] = np.nan
l=0
lab=['a)','b)']
for k in range(2):
    ax=fig.add_subplot(gs[l,k],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=2, edgecolor='black')
    ax.contour(Zlon,Zlat,ZB,levels=elevation_levels,colors='purple',linewidths=0.35)
    lc=ax.contourf(LON,LAT,gridmap[what[k]]*100,levels=pvr_levels,cmap=cmap,norm=norm,zorder=1,extend='max')
    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
    ax.text(0.025,0.9,lab[k+l*2],fontsize=10,fontweight='bold',transform=ax.transAxes)

cax=plt.axes([0.775,0.1,0.01,0.4])
plt.subplots_adjust(wspace=0,hspace=0,right=0.775,bottom=0.1,top=0.5)
cbar=plt.colorbar(lc, ticks=pvr_levels,cax=cax)
cbar.ax.set_yticklabels(np.append(r'%',np.round(ticklabels[1:],4)))
figname = psave +'difference-gridmap' + '-PVedge-' + str(PVedge) + '.png'
#fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close('all')

