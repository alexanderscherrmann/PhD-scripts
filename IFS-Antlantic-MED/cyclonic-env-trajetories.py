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
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import pickle
import os
import matplotlib.patches as patch
import xarray as xr

import cartopy
import matplotlib.gridspec as gridspec
from matplotlib.legend_handler import HandlerLineCollection


class HandlerColorLineCollection(HandlerLineCollection):
    def create_artists(self, legend, artist ,xdescent, ydescent,
                        width, height, fontsize,trans):
        x = np.linspace(0,width,self.get_numpoints(legend)+1)
        y = np.zeros(self.get_numpoints(legend)+1)+height/2.-ydescent
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=artist.cmap,
                     transform=trans)
        lc.set_array(x)
        lc.set_linewidth(artist.get_linewidth())
        return [lc]


pload = '/home/ascherrmann/010-IFS/traj/MED/use/'

f = open(pload + 'PV-data-MEDdPSP-100-ZB-800PVedge-0.3-400.txt','rb')
PVdata = pickle.load(f)
f.close()

dipv = PVdata['dipv']
dit = PVdata['dit']
datadi = PVdata['rawdata']

adv = np.array([])
cyc = np.array([])
env = np.array([])
c = 'cyc'
e = 'env'

f = open('/home/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','rb')
locdata =  pickle.load(f)
f.close()

cyctrajid = 1256
envtrajid = 356#274#304
#bothtrajid = 356#1440


trajids = np.array([cyctrajid,envtrajid])#,bothtrajid])

#fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
colors = ['red','blue']

#ax.coastlines()

fig=plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(ncols=1, nrows=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)

LON = np.linspace(-180,180,901)
LAT = np.linspace(0,90,226)
lon = np.array([])
lat = np.array([])
for lo,la in zip(locdata['DEC17'][73]['clon'],locdata['DEC17'][73]['clat']):
    lon = np.append(lon,np.mean(lo))
    lat = np.append(lat,np.mean(la))


mat = abs(locdata['DEC17'][73]['hzeta'][0]).astype(int)

i = np.where(datadi['20171214_02-073']['PV'][:,0]>=0.75)[0]

for x in i:
 #   if (dipv['20171214_02-073']['env']['deltaPV'][x,0]/datadi['20171214_02-073']['PV'][x,0])>0.8:
    if (dipv['20171214_02-073']['env']['deltaPV'][x,0]>=0.3) and (dipv['20171214_02-073']['env']['deltaPV'][x,0]>=0.5):
        print(x,dit['20171214_02-073']['cyc'][x,:],dipv['20171214_02-073']['env']['deltaPV'][x,0])
#    k = np.where(dipv['20171214_02-073']['env']['deltaPV'][x,0]>=0.5)[0][0]
#    if k>=30:
#    print(,x,datadi['20171214_02-073']['PV'][x,0],dit['20171214_02-073']['cyc'][x,:])




cmap1 = ListedColormap(['lightcoral','red'])
cmap2 = ListedColormap(['deepskyblue','blue'])
#cmap2 = ListedColormap(['silver','slategray'])

colo = ['lightcoral','deepskyblue']
cmap = [cmap1,cmap2]#,cmap3]
norm = BoundaryNorm([0, 0.5, 1], cmap1.N)

linewidth = 2
alpha = 1

for q,k in enumerate(trajids):
#    ax.plot(datadi['20171214_02-073']['lon'][k],datadi['20171214_02-073']['lat'][k],color=colors[q])
    for x in range(0,49):
        seg = helper.make_segments(datadi['20171214_02-073']['lon'][k,:],datadi['20171214_02-073']['lat'][k,:])
        z = dit['20171214_02-073']['cyc'][k,:]
        lc = mcoll.LineCollection(seg, array=z, cmap=cmap[q], norm=norm, linewidth=linewidth, alpha=alpha)        
        ax.add_collection(lc)
    ax.scatter(datadi['20171214_02-073']['lon'][k,31],datadi['20171214_02-073']['lat'][k,31],color=colo[q],marker='o',s=50,zorder=50)

ax.text(0.05, 0.95, 'b)', transform=ax.transAxes,fontsize=16, fontweight='bold', va='top')
ax.plot(LON[lon.astype(int)],LAT[lat.astype(int)],color='k',linewidth=2)
ax.scatter(LON[lon.astype(int)][mat-29],LAT[lat.astype(int)][mat-29],color='k',s=50)

#for u in range(0,mat+1):
#    circ2 = patch.Circle((LON[lon.astype(int)][u],LAT[lat.astype(int)][u]),helper.convert_radial_distance_to_lon_lat_dis(400),edgecolor='k',facecolor='none',linestyle='-',linewidth=.5,zorder=10.,alpha=0.5)
#    ax.add_patch(circ2)


lonticks = np.arange(10,35.1,5)
latticks = np.arange(
        30,50.1,5)

ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
ax.set_yticks(latticks, crs=ccrs.PlateCarree());
ax.set_xticklabels(labels=lonticks,fontsize=10)
ax.set_yticklabels(labels=latticks,fontsize=10)

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())


pf = xr.open_dataset('/home/ascherrmann/010-IFS/data/DEC17/P20171212_21')
h22 = ax.contour(np.linspace(-180,180,901),np.linspace(0,90,226),pf.SLP.values[0,0,:,:],levels=np.arange(990,1021,5),colors='purple',animated=True,linewidths=1., alpha=1)
labsl={}
strs = ['990','995','1000','1005','1010','1015','1020']

circ2 = patch.Circle((LON[lon[mat-29].astype(int)],LAT[lat[mat-29].astype(int)]),helper.convert_radial_distance_to_lon_lat_dis(400),edgecolor='k',facecolor='none',linestyle='-',linewidth=1.5,zorder=10.,alpha=1)
#circ1 = patch.Circle((LON[lon[mat].astype(int)],LAT[lat[mat].astype(int)]),helper.convert_radial_distance_to_lon_lat_dis(200),edgecolor='k',facecolor='none',linestyle='--',linewidth=1.5,zorder=10.,alpha=1)

ax.add_patch(circ2)
#ax.add_patch(circ1)
ax.set_extent([5,30,30,45], ccrs.PlateCarree())
for l,s in zip(h22.levels[:],strs):
    labsl[l] = s
plt.clabel(h22,inline=True, fontsize=10, fmt='%d',manual=True)


#circ2 = patch.Circle((np.mean(np.unique(datadi['20171214_02-073']['lon'][:,0])),np.mean(np.unique(datadi['20171214_02-073']['lat'][:,0]))),helper.convert_radial_distance_to_lon_lat_dis(400),edgecolor='k',facecolor='none',linestyle='-',linewidth=1.5,zorder=10.,alpha=1)
#ax.add_patch(circ2)
#ax.set_extent([5,40,25,50], ccrs.PlateCarree())


fig.savefig('/home/ascherrmann/010-IFS/single-traj-explanation-b-2-400.png',dpi=300,bbox_inches="tight")
plt.close('all')

fig,axes = plt.subplots(1,1)#,sharex=True)
#plt.subplots_adjust(left=0.1,bottom=None,top=None,right=None,hspace=0,wspace=0.0)
#axes = axes.flatten()
ax =axes#[0]
ax.set_xticks(ticks=np.arange(-48,1,6))
ax.set_ylabel('PV [PVU]')
ax.tick_params(labelright=False,right=True)
ax.text(0.05, 0.95, 'b)',transform=ax.transAxes,fontsize=16, fontweight='bold', va='top')
for q,k in enumerate(trajids):
    ax.plot(datadi['20171214_02-073']['time'][0],datadi['20171214_02-073']['PV'][k],color=colors[q])


#ax = axes[1]
#ax.text(0.05, 0.95, 'c)',transform=ax.transAxes,fontsize=16, fontweight='bold', va='top')
#for q,k in enumerate(trajids):
#    ax.plot(datadi['20171214_02-073']['time'][0],dit['20171214_02-073']['cyc'][k],color=colors[q])
#
ax.set_xlabel('time from mature stage [h]')
#ax.set_ylabel('inside cyclone')
ax.set_xticks(ticks=np.arange(-48,1,6))
ax.tick_params(labelright=False,right=True)
ax.set_xlim(-48,0)
fig.savefig('/home/ascherrmann/010-IFS/inside-cyclone-400.png',dpi=300,bbox_inches="tight")
plt.close('all')





ditc = dit['20171214_02-073']['cyc']
dite = dit['20171214_02-073']['env']

fig,axes = plt.subplots(2,1,sharex=True)
plt.subplots_adjust(left=0.1,bottom=None,top=None,right=None,hspace=0,wspace=0.0)
axes = axes.flatten()
#ax =axes[0]
#ax.set_xticks(ticks=np.arange(-48,1,6))
#ax.set_ylabel('PV [PVU]')
#ax.tick_params(labelright=False,right=True)
#ax.text(0.02, 0.95, 'c)',transform=ax.transAxes,fontsize=16, fontweight='bold', va='top')

#for q,k in enumerate(trajids[:1]):
#    deltaPV = np.zeros(49)
#    deltaPV[:-1] = datadi['20171214_02-073']['PV'][k,:-1]-datadi['20171214_02-073']['PV'][k,1:]
##    ax.plot(datadi['20171214_02-073']['time'][0],deltaPV,color='k')
#    ax.plot(datadi['20171214_02-073']['time'][0],datadi['20171214_02-073']['PV'][k],color='grey')
#    ax.plot(datadi['20171214_02-073']['time'][0],np.flip(np.cumsum(np.flip(deltaPV))),color='k')
#    ax.plot(datadi['20171214_02-073']['time'][0],np.flip(np.cumsum(np.flip(deltaPV * ditc[k]))),color='red')
#    ax.plot(datadi['20171214_02-073']['time'][0],np.flip(np.cumsum(np.flip(deltaPV * dite[k]))),color='lightcoral',ls=':')

#ax = axes[1]

cmap1 = ListedColormap(['red','blue'])
cmap2 = ListedColormap(['lightcoral','deepskyblue'])

lc1 = mcoll.LineCollection(seg*10, array=z, cmap=cmap1, norm=norm, linewidth=linewidth, alpha=alpha)
lc2 = mcoll.LineCollection(seg*10, array=z, cmap=cmap2, norm=norm, linewidth=linewidth, alpha=alpha)
colors1 = ['red','blue']
colors2 = ['lightcoral','deepskyblue']

#ax.text(0.02, 0.95, 'd)',transform=ax.transAxes,fontsize=16, fontweight='bold', va='top')

lab = ['c)','d)']
for q,k in enumerate(trajids):
    al, = ax.plot([],[],ls='-',color='grey')
    bl, = ax.plot([],[],ls='-',color='k')

    if q==1:
     ax.add_collection(lc1)
     ax.add_collection(lc2)

    ax = axes[q]
    deltaPV = np.zeros(49)
    deltaPV[:-1] = datadi['20171214_02-073']['PV'][k,:-1]-datadi['20171214_02-073']['PV'][k,1:]
    ax.plot(datadi['20171214_02-073']['time'][0],datadi['20171214_02-073']['PV'][k],color='grey')
    ax.plot(datadi['20171214_02-073']['time'][0],np.flip(np.cumsum(np.flip(deltaPV))),color='k')
    ax.plot(datadi['20171214_02-073']['time'][0],np.flip(np.cumsum(np.flip(deltaPV * ditc[k]))),color=colors1[q])
    ax.plot(datadi['20171214_02-073']['time'][0],np.flip(np.cumsum(np.flip(deltaPV * dite[k]))),color=colors2[q],linestyle=':')
    ax.text(-0.1, 0.95, lab[q],transform=ax.transAxes,fontsize=12, fontweight='bold', va='top')

axes[0].legend([al,bl,lc1,lc2],['PV','apv','apvc','apve'],handler_map={lc1: HandlerColorLineCollection(numpoints=2),lc2: HandlerColorLineCollection(numpoints=2)},frameon=False,loc='upper left')
axes[0].set_ylabel('PV [PVU]')
#axes[1].legend([lc1,lc2],['apvc','apve'],handler_map={lc1: HandlerColorLineCollection(numpoints=2),lc2: HandlerColorLineCollection(numpoints=2)},loc='upper left',frameon=False)

#ax.plot(datadi['20171214_02-073']['time'][0],avc/3,color='k')
#ax.plot(datadi['20171214_02-073']['time'][0],ave/3,color='k',linestyle=':')

ax.set_xlabel('time from mature stage [h]')
ax.set_ylabel('PV [PVU]')
ax.set_xticks(ticks=np.arange(-48,1,6))
ax.tick_params(labelright=False,right=True)
ax.set_xlim(-48,0)
fig.savefig('/home/ascherrmann/010-IFS/edu-env-cycPV-300.png',dpi=300,bbox_inches="tight")
plt.close('all')
