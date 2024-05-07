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
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def colbar(cmap,minval,maxval,nlevels,levels):
    maplist = [cmap(i) for i in range(cmap.N)]
    newmap = ListedColormap(maplist)
    norm = BoundaryNorm(levels,cmap.N)
    return newmap, norm

p = '/home/ascherrmann/009-ERA-5/'

traced = np.array([])
for d in os.listdir(p):
    if(d.startswith('traced-vars-20')):
            traced = np.append(traced,d)
traced = np.sort(traced)

LON=np.linspace(-180,180,721)
LAT=np.linspace(-90,90,361)

PVdata = np.loadtxt(p + 'manos-test-data.txt')

rdis = 400
labs = helper.traced_vars_ERA5()
traced = np.array([])
traj = np.array([])
H = 47
a = 1

for d in os.listdir(p):
    if(d.startswith('traced-vars-2full')):
            traced = np.append(traced,d)
traced = np.sort(traced)

maxv = 0.61
minv =-0.6
pvr_levels = np.arange(minv,maxv,0.15)

ap = plt.cm.seismic
cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels),pvr_levels)
for k in range(75,150):
    cmap.colors[k] = np.array([189/256, 195/256, 199/256, 1.0])

maxv = 0.51
minv = -0.5
reslvl = np.arange(minv,maxv,0.1)
ap = plt.cm.seismic
cmap2, norm2 = colbar(ap,minv,maxv,len(reslvl),reslvl)

#for k in range(75,150):
#    cmap2.colors[k]=np.array([189/256, 195/256, 199/256, 1.0])


alpha=1.
linewidth=.2
ticklabels=pvr_levels

figg, axg = plt.subplots(1, 1)
figgg,axgg = plt.subplots(1, 1)
figggg,axggg = plt.subplots(1, 1)

for q,t in enumerate(traced[:]):
    data = np.loadtxt(p + t)
    ID = int(t[-10:-4])

    tralon = data[:,1].reshape(-1,H+1)
    tralat = data[:,2].reshape(-1,H+1)

    pvr = np.zeros(tralon.shape)
    for z in ['pvf','pvt']:
        pvr +=data[:,np.where(labs==z)].reshape(-1,H+1)
    dd = np.where(PVdata[:,-2]==ID)[0]

    clat2 = np.where(LAT==PVdata[dd,1])[0]
    clon2 = np.where(LON==PVdata[dd,0])[0]

    latc = LAT[clat2]
    lonc = LON[clon2]

    ran=int(np.min(np.where(np.linspace(0,2*np.pi*6370,721)>2000)))

    latu = np.arange(clat2-ran,clat2+ran+1,1)
    lonu = np.arange(clon2-ran,clon2+ran+1,1)
    minlatc = np.min(latu)
    minlonc = np.min(lonu)

    pltlat =np.linspace(-90,90,361)[latu]
    pltlon = np.linspace(-180,180,721)[lonu]

    minlatc = np.min(latu)
    maxlatc = np.max(latu)
    minpltlatc = pltlat[0]
    maxpltlatc = pltlat[-1]
    minlonc = np.min(lonu)
    maxlonc = np.max(lonu)
    minpltlonc = pltlon[0]
    maxpltlonc = pltlon[-1]

    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.coastlines()
    for q in range(len(tralon[:,0])):
        seg = helper.make_segments(tralon[q,:],tralat[q,:])
        z = pvr[q,:]
        lc = mcoll.LineCollection(seg, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
        ax=plt.gca()
        ax.add_collection(lc)

    lonticks=np.arange(minpltlonc, maxpltlonc,5)
    latticks=np.arange(minpltlatc, maxpltlatc,5)

    ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
    ax.set_yticks(latticks, crs=ccrs.PlateCarree());
    ax.set_xticklabels(labels=lonticks,fontsize=8)
    ax.set_yticklabels(labels=latticks,fontsize=8)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())

    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(lc, ticks=pvr_levels,cax=cbax)
    
    func=resize_colorbar_vert(cbax, ax, pad=0.01, size=0.02)
    fig.canvas.mpl_connect('draw_event', func)
    
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel('PVU/h',fontsize=8)
    cbar.ax.set_xticklabels(ticklabels)

    circ2 = patch.Circle((lonc,latc),helper.convert_radial_distance_to_lon_lat_dis(400),edgecolor='k',facecolor='none',linestyle='--',linewidth=.5,zorder=10.)
    circ1 = patch.Circle((lonc,latc),helper.convert_radial_distance_to_lon_lat_dis(200),edgecolor='k',linestyle='-',facecolor='None',linewidth=.5,zorder=10.)
    
    ax.add_patch(circ1)
    ax.add_patch(circ2)    

    figname = p + 'traj-2full-pvr-mature-' + t[-25:-4] + '.png'
    fig.savefig(figname,dpi=300,bbox_inches="tight")
    plt.close()


    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.coastlines()

    res = np.zeros(pvr.shape)
    PV = data[:,-1].reshape(-1,H+1)
    res[:,1:-1] = np.flip(np.cumsum(np.flip(pvr[:,1:-1],axis=1),axis=1),axis=1) - (PV[:,1:-1]-PV[:,-1].reshape(-1,1))
    res[:,0] = res[:,1]

    avres = np.mean(res,axis=0)
    axg.plot(np.flip(np.arange(-47,1)),avres,color='k')
    axgg.scatter(np.mean(PV[:,1]),avres[1],color='k',marker='x')
    axggg.scatter(np.mean(PV[:,-1]),avres[1],color='k',marker='x')
    for q in range(len(tralon[:,0])):
        seg = helper.make_segments(tralon[q,:],tralat[q,:])
        z = res[q,:]
        lc = mcoll.LineCollection(seg, array=z, cmap=cmap2, norm=norm2, linewidth=linewidth, alpha=alpha)
        ax=plt.gca()
        ax.add_collection(lc)

    lonticks=np.arange(minpltlonc, maxpltlonc,5)
    latticks=np.arange(minpltlatc, maxpltlatc,5)

    ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
    ax.set_yticks(latticks, crs=ccrs.PlateCarree());
    ax.set_xticklabels(labels=lonticks,fontsize=8)
    ax.set_yticklabels(labels=latticks,fontsize=8)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())

    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(lc, ticks=reslvl,cax=cbax)

    func=resize_colorbar_vert(cbax, ax, pad=0.01, size=0.02)
    fig.canvas.mpl_connect('draw_event', func)

    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel('res. [PVU]',fontsize=8)
    cbar.ax.set_xticklabels(reslvl)

    circ2 = patch.Circle((lonc,latc),helper.convert_radial_distance_to_lon_lat_dis(400),edgecolor='k',facecolor='none',linestyle='--',linewidth=.5,zorder=10.)
    circ1 = patch.Circle((lonc,latc),helper.convert_radial_distance_to_lon_lat_dis(200),edgecolor='k',linestyle='-',facecolor='None',linewidth=.5,zorder=10.)

    ax.add_patch(circ1)
    ax.add_patch(circ2)

    figname = p + 'traj-2full-residual-mature-' + t[-25:-4] + '.png'
    fig.savefig(figname,dpi=300,bbox_inches="tight")
    plt.close()


axg.set_xlabel('time to mature stage [h]')
axg.set_ylabel('residual [PVU]')
axg.set_xlim(-47,0)
axg.set_ylim(-1,1)

figg.savefig(p + '2full-residualevol.png',dpi=300,bbox_inches="tight")
plt.close()

axgg.set_xlabel('PV end [PVU]')
axgg.set_ylabel('residual [PVU]')
axgg.set_ylim(-1,1)
figgg.savefig(p + '2full-residual-PVend-scatter.png',dpi=300,bbox_inches="tight")
plt.close()

axggg.set_xlabel('PV start [PVU]')
axggg.set_ylabel('residual [PVU]')
axggg.set_ylim(-1,1)
axggg.set_xlim(-2,2)
figggg.savefig(p + '2full-residual-PVstart-scatter.png',dpi=300,bbox_inches="tight")
plt.close()
