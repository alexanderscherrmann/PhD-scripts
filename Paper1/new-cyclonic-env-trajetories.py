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

###
### for illustrative purposes and know IDS of trajectories use old data
###
pload = '/atmosdyn2/ascherrmann/010-IFS/traj/MED/use/'

f = open(pload + 'PV-data-MEDdPSP-100-ZB-800PVedge-0.3-400.txt','rb')
PVdata = pickle.load(f)
f.close()

dipv = PVdata['dipv']
dit = PVdata['dit']
ORO = PVdata['oro']
datadi = PVdata['rawdata']

adv = np.array([])
cyc = np.array([])
env = np.array([])
c = 'cyc'
e = 'env'
o = 'oro'
n = 'noro'

NORO = xr.open_dataset('/home/ascherrmann/scripts/ERA5-utils/NORO')
ZB = NORO['ZB'].values[0]
Zlon = NORO['lon']
Zlat = NORO['lat']
Emaxv = 3000
Eminv = 800
elevation_levels = np.arange(Eminv,Emaxv,400)

f = open('/atmosdyn2/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','rb')
locdata =  pickle.load(f)
f.close()

cyctrajid = 1256
envtrajid = 356
orotrajid = 1452

trajids = np.array([cyctrajid,envtrajid,orotrajid])

colors = ['red','blue','saddlebrown']

###
# FIG a)
###

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

#for x in i:
#    if ORO['20171214_02-073']['env']['APVTOT'][x,0]>0 and dipv['20171214_02-073']['env']['APVTOT'][x,0]>0:
#        print(x)


cmap1 = ListedColormap(['lightcoral','red'])
cmap2 = ListedColormap(['deepskyblue','blue'])
cmap3 = ListedColormap(['silver','slategray'])

#colo = ['lightcoral','deepskyblue','silver']
cmap = [cmap1,cmap2,cmap3]
norm = BoundaryNorm([0, 0.5, 1], cmap1.N)

linewidth = 2
alpha = 1

for q,k in enumerate(trajids):
#    ax.plot(datadi['20171214_02-073']['lon'][k],datadi['20171214_02-073']['lat'][k],color=colors[q])
    ax.plot(datadi['20171214_02-073']['lon'][k,:],datadi['20171214_02-073']['lat'][k,:],color=colors[q],linewidth=2.)
#    for x in range(0,49):
#        seg = helper.make_segments(datadi['20171214_02-073']['lon'][k,:],datadi['20171214_02-073']['lat'][k,:])
#        z = dit['20171214_02-073']['cyc'][k,:]
#        lc = mcoll.LineCollection(seg, array=z, cmap=cmap[q], norm=norm, linewidth=linewidth, alpha=alpha)        
#        ax.add_collection(lc)
#    ax.scatter(datadi['20171214_02-073']['lon'][k,31],datadi['20171214_02-073']['lat'][k,31],color=colo[q],marker='o',s=50,zorder=50)


ax.text(0.025, 0.975, '(a)', transform=ax.transAxes,fontsize=16, va='top')
ax.plot(LON[lon.astype(int)],LAT[lat.astype(int)],color='k',linewidth=2)
ax.scatter(LON[lon.astype(int)][mat],LAT[lat.astype(int)][mat],color='k',s=50)
#ax.scatter(datadi['20171214_02-073']['lon'][k,30],datadi['20171214_02-073']['lat'][k,30],color='orange',marker='s',s=50)

lonticks = np.arange(10,35.1,5)
latticks = np.arange(30,50.1,5)

ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
ax.set_yticks(latticks, crs=ccrs.PlateCarree());
ax.set_xticklabels(labels=lonticks,fontsize=10)
ax.set_yticklabels(labels=latticks,fontsize=10)

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.contour(Zlon,Zlat,ZB,levels = elevation_levels,colors='purple',linewidths=0.5,alpha=1,zorder=10000)

pf = xr.open_dataset('/atmosdyn2/ascherrmann/010-IFS/data/DEC17/P20171214_02')
#h22 = ax.contour(np.linspace(-180,180,901),np.linspace(0,90,226),pf.SLP.values[0,0,:,:],levels=np.arange(990,1021,5),colors='grey',animated=True,linewidths=1., alpha=1)
labsl={}
strs = ['990','995','1000','1005','1010','1015','1020']

circ2 = patch.Circle((LON[lon[mat].astype(int)],LAT[lat[mat].astype(int)]),helper.convert_radial_distance_to_lon_lat_dis(400),edgecolor='k',facecolor='none',linestyle='-',linewidth=1.5,zorder=10.,alpha=1)
circ1 = patch.Circle((LON[lon[mat].astype(int)],LAT[lat[mat].astype(int)]),helper.convert_radial_distance_to_lon_lat_dis(200),edgecolor='k',facecolor='none',linestyle='--',linewidth=1.5,zorder=10.,alpha=1)

ax.add_patch(circ2)
ax.add_patch(circ1)
ax.set_extent([5,30,30,50], ccrs.PlateCarree())
#for l,s in zip(h22.levels[:],strs):
#    labsl[l] = s
#plt.clabel(h22,inline=True, fontsize=10, fmt='%d',manual=True)


fig.savefig('/atmosdyn2/ascherrmann/paper/cyc-env-PV/review/inside-cyclone-explaination.png',dpi=300,bbox_inches="tight")
plt.close(fig)


####
#FIG b,c,d)
#####

ditc=dit['20171214_02-073'][c]
dite=dit['20171214_02-073'][e]
diteno = (dit['20171214_02-073'][o]+1)%2
diteo = dit['20171214_02-073'][o]

fig,axes = plt.subplots(3,1,sharex=True,figsize=(8,6))
plt.subplots_adjust(left=0.1,bottom=None,top=None,right=None,hspace=0,wspace=0.0)
axes = axes.flatten()
colors1 = ['red','blue','saddlebrown']
lab = ['(b)','(c)','(d)']


cmap1 = ListedColormap(colors1)
norm = BoundaryNorm([0, 1./3., 2./3., 1], cmap1.N)
alpha = 1
seg=[((1,1),(2,2),(3,3))]
z=np.array([0.2,0.5,0.8])

lc1 = mcoll.LineCollection(seg*10, array=z, cmap=cmap1, norm=norm, linewidth=linewidth, alpha=alpha)

a1,=axes[0].plot([],[],ls=' ',marker='o',color='k')
a2,=axes[0].plot([],[],ls='-',color='k')
a3,=axes[0].plot([],[],ls=':',color='k')
a4,=axes[0].plot([],[],ls=':',color='grey')

#axes[2].plot([],[],ls='-',color='grey')
#axes[0].plot([],[],ls='-',color='grey')

for q,k in enumerate(trajids):
    ax = axes[q]
    deltaPV = np.zeros(49)
    deltaPV[:-1] = datadi['20171214_02-073']['PV'][k,:-1]-datadi['20171214_02-073']['PV'][k,1:]
    ax.plot(datadi['20171214_02-073']['time'][0],datadi['20171214_02-073']['PV'][k],color=colors1[q])
    ax.plot(datadi['20171214_02-073']['time'][0],np.flip(np.cumsum(np.flip(deltaPV * ditc[k]))),color='k',linestyle='-')
    #ax.plot(datadi['20171214_02-073']['time'][0],np.flip(np.cumsum(np.flip(deltaPV * dite[k]))),color='k',linestyle=':')
    ax.plot(datadi['20171214_02-073']['time'][0],np.flip(np.cumsum(np.flip(deltaPV * diteno[k]*dite[k]))),color='k',linestyle=':')
    ax.plot(datadi['20171214_02-073']['time'][0],np.flip(np.cumsum(np.flip(deltaPV * diteo[k]*dite[k]))),color='grey',linestyle=':')


    ax.text(-0.105, 0.95, lab[q],transform=ax.transAxes,fontsize=16,va='top')

    cyids = np.where(ditc[k]==1)[0]
    envids = np.where(dite[k]==1)[0]
    ax.scatter(datadi['20171214_02-073']['time'][0,cyids],datadi['20171214_02-073']['PV'][k,cyids],color=colors1[q],marker='o',s=30.)
    ax.set_ylim(-0.7,2.75)
    ax.set_ylabel('PV [PVU]')
    ax.set_yticks(ticks=[0,0.5,1,1.5,2,2.5])
    if q==1:
        ax.set_ylim(-2.5,2.5)
        ax.set_yticks(ticks=[-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5])
    if q==2:
        ax.set_ylim(-1,2.5)
        ax.set_yticks(ticks=np.arange(-1,3,0.5))

    ax.tick_params(labelright=False,right=True)
    ax.set_xticks(ticks=np.arange(-48,1,6))
    ax.set_xlim(-48,0)

#plot oro PV of oro trajectory and load it before
f=open('/atmosdyn2/ascherrmann/010-IFS/ctraj/MED/use/PV-data-MEDdPSP-100-ZB-800PVedge-0.3-400-correct-distance.txt','rb')
newdata =pickle.load(f)
f.close()
mID = 829

#ax.plot(datadi['20171214_02-073']['time'][0],newdata['oro']['20171214_02-073']['env']['deltaPV'][mID],color='grey',zorder=0)

axes[0].legend([lc1,a1,a2,a3,a4],['PV','inside cyclone','APVC','APVENO','APVEO'],loc='upper left',frameon=False,handler_map={lc1: HandlerColorLineCollection(numpoints=3)})
#axes[2].legend(['apvo'],loc='upper left',frameon=False)
ax.set_xlabel('time from mature stage [h]')
#ax.set_ylabel('PV [PVU]')
fig.savefig('/atmosdyn2/ascherrmann/paper/cyc-env-PV/review/noro-edu-env-cycPV-400.png',dpi=300,bbox_inches="tight")
plt.close('all')
