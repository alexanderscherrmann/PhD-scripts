import numpy as np
import os
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')

from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import helper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patch
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle
import xarray as xr

import cartopy
import matplotlib.gridspec as gridspec
import functools

def colbar(cmap,minval,maxval,nlevels,levels):
    maplist = [cmap(i) for i in range(cmap.N)]
    newmap = ListedColormap(maplist)
    norm = BoundaryNorm(levels,cmap.N)
    return newmap, norm

pload = '/atmosdyn2/ascherrmann/009-ERA-5/MED/cases/'

f = open(pload + 'zorbas-data.txt','rb')
cdata = pickle.load(f)
f.close()

tdata = np.loadtxt(pload + 'trajectories-mature-20180928_05-ID-524945.txt')

NORO = xr.open_dataset('/home/ascherrmann/scripts/ERA5-utils/NORO')
ZB = NORO['ZB'].values[0]
Zlon = NORO['lon']
Zlat = NORO['lat']
Emaxv = 3000
Eminv = 800
elevation_levels = np.arange(Eminv,Emaxv,400)

rdis = 400
H = 48
a = 1

maxv = 0.61
minv =-0.6
pvr_levels = np.arange(minv,maxv,0.15)

P = tdata[:,-2].reshape(-1,49)
PV = tdata[:,-1].reshape(-1,49)

idp = np.where((P[:,0]<=975) & (P[:,0]>=700) & (PV[:,0]>=0.75))[0]

ap = plt.cm.BrBG
cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels),pvr_levels)
for k in range(75,150):
    cmap.colors[k] = np.array([189/256, 195/256, 199/256, 1.0])

alpha=1.
linewidth=1.0
ticklabels=pvr_levels


tralon = tdata[:,1].reshape(-1,49) 
tralat = tdata[:,2].reshape(-1,49)
tralon = tralon[idp]
tralat = tralat[idp]

deltaPV = np.zeros(tralon.shape)
deltaPV[:,1:] = PV[idp,:-1]-PV[idp,1:]

pvr = deltaPV   

latc = cdata['lat']
lonc = cdata['lon']
latm = cdata['lat'][abs(cdata['hourstoSLPmin'][0]).astype(int)]
lonm = cdata['lon'][abs(cdata['hourstoSLPmin'][0]).astype(int)]
time0=abs(cdata['hourstoSLPmin'][0])

minpltlatc = np.round(latm-np.floor(helper.convert_radial_distance_to_lon_lat_dis(800)),0)
minpltlonc = np.round(lonm-np.floor(helper.convert_radial_distance_to_lon_lat_dis(800)),0)

maxpltlatc = np.round(latm+np.round(helper.convert_radial_distance_to_lon_lat_dis(2000),0),0)
maxpltlonc = np.round(lonm+np.round(helper.convert_radial_distance_to_lon_lat_dis(2000),0),0)

import cartopy.feature as cfeature


for hours in range(0,49):
    if (hours+1)<(49-time0):
        continue
    fig=plt.figure(figsize=(8,6))
    titles=['cyclogenesis', '48 h after cyclogenesis']
    gs = gridspec.GridSpec(ncols=1, nrows=1)# figure=fig)
    ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)
    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
    #bfi = xr.open_dataset('/atmosdyn2/era5/cdf/2018/09/B20180928_04')
    
    #fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))#subplot_kw=dict(projection=ccrs.PlateCarree()))
    #ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc]) 
    #ax.add_feature(cfeature.LAND)
    #plt.show()
    #ax.coastlines()
    
    for q in range(len(tralon[:,0])):
        if hours!=0:
            seg = helper.make_segments(tralon[q,-1*(hours+1):-(hours-1)],tralat[q,-1*(hours+1):-(hours-1)])
            z = pvr[q,-1*(hours+1):-(hours-1)]
        else:
            seg = helper.make_segments(tralon[q,-1*(hours+1):],tralat[q,-1*(hours+1):])
            z = pvr[q,-1*(hours+1):]
        lc = mcoll.LineCollection(seg, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
        ax=plt.gca()
        ax.add_collection(lc)
    

    #ax.contourf(Zlon,Zlat,ZB,levels=
    #con = ax.countour(np.arange(-180,180,0.5),np.arange(-90,90,0.5),bfi.MSL.values[0]/100)
    #ax.clabel(con,inline=True,fontsize=10,fmt='%d',manual=True)
    
    #ax.scatter(lonc[0],latc[0],marker='x',color='k',s=40)
    lonticks=np.arange(minpltlonc, maxpltlonc,5)
    latticks=np.arange(minpltlatc, maxpltlatc,5)
    
    ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
    ax.set_yticks(latticks, crs=ccrs.PlateCarree());
    ax.set_xticklabels(labels=lonticks,fontsize=8)
    ax.set_yticklabels(labels=latticks,fontsize=8)
    
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    #ax.text(0.03, 0.95,'(a)', transform=ax.transAxes,fontsize=16,va='top')
    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(lc, ticks=np.round(pvr_levels,2),cax=cbax)
    
    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
    fig.canvas.mpl_connect('draw_event', func)
    
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel('PVU h$^{-1}$',fontsize=10)
    cbar.ax.set_yticklabels(np.round(ticklabels,2))
    
    if hours+1>=(49-time0):
        print(time0,hours,time0-49+hours+1)
    #    ax.plot(lonc,latc,color='k')
        ax.scatter(lonc[(time0-49+hours+1).astype(int)],latc[(time0-49+hours+1).astype(int)],marker='o',color='k',s=40,zorder=100)
        circ1 = patch.Circle((lonc[(time0-49+hours+1).astype(int)],latc[(time0-49+hours+1).astype(int)]),helper.convert_radial_distance_to_lon_lat_dis(400),edgecolor='k',linestyle='-',facecolor='None',linewidth=1.5,zorder=10.)
    
        ax.add_patch(circ1)
    if hours==48:
        circ2 = patch.Circle((lonm,latm),helper.convert_radial_distance_to_lon_lat_dis(200),edgecolor='k',facecolor='none',linestyle='--',linewidth=2,zorder=10.)
        ax.add_patch(circ2)  
    
    figname = 'traj-pvr-zorbas_mature-%02d.png'%hours
    fig.savefig('/home/ascherrmann/defense/' + figname,dpi=300,bbox_inches="tight")
    plt.close('all')

x = cdata['hourstoSLPmin']
da = cdata['dates']

#fig, ax = plt.subplots(1, 1)
#ax.plot(cdata['hourstoSLPmin'],cdata['SLP'],color='k')
#ax.set_xlim(x[0],x[-1])
#ax.set_ylim(994,1010)
#
#ax.set_xticks(ticks=np.array([x[abs(x[0]).astype(int)-24],x[abs(x[0]).astype(int)-18],x[abs(x[0]).astype(int)-12],x[abs(x[0]).astype(int)-6],x[abs(x[0]).astype(int)],x[abs(x[0]).astype(int)+6],x[abs(x[0]).astype(int)+12],x[abs(x[0]).astype(int)+18],x[abs(x[0]).astype(int)+24],x[abs(x[0]).astype(int)+30],x[abs(x[0]).astype(int)+36],x[-1]]))
#ax.set_xticklabels(labels=np.array([da[abs(x[0]).astype(int)-24],da[abs(x[0]).astype(int)-18],da[abs(x[0]).astype(int)-12],da[abs(x[0]).astype(int)-6],da[abs(x[0]).astype(int)],da[abs(x[0]).astype(int)+6],da[abs(x[0]).astype(int)+12],da[abs(x[0]).astype(int)+18],da[abs(x[0]).astype(int)+24],da[abs(x[0]).astype(int)+30],da[abs(x[0]).astype(int)+36],da[-1]]))
#plt.xticks(rotation=90)
#
#ax.set_ylabel('SLP [hPa]')
#ax.text(0.03, 0.95,'c)', transform=ax.transAxes,fontsize=12, fontweight='bold',va='top')
#fig.savefig(pload+'zorbas-SLP.png',dpi=300,bbox_inches="tight")
#plt.close('all')
