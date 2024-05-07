import numpy as np
from netCDF4 import Dataset as ds
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy
from matplotlib.colors import BoundaryNorm
import wrf
import sys
import cmocean

sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
from colormaps import PV_cmap2

from dypy.intergrid import Intergrid
from mpl_toolkits.basemap import Basemap
import helper

seasons = ['DJF','MAM','JJA','SON']
pvpos = [[93,55],[95,56],[140,73],[125,72]]
amps = [0.7,1.4,2.1,2.8]

pvcmap,pvlevels,pvnorm,pvticklabels=PV_cmap2()
ucmap,ulevels = cmocean.cm.tempo,np.arange(10,65,5)
unorm = BoundaryNorm(ulevels,ucmap.N)
PSlevel = np.arange(975,1031,5)
dis = 1500

dlat = helper.convert_radial_distance_to_lon_lat_dis_new(dis,0)

wrfd = '/atmosdyn2/ascherrmann/013-WRF-sim/'
pappath = '/atmosdyn2/ascherrmann/015-CESM-WRF/'

fig=plt.figure(figsize=(10,12))
gs = gridspec.GridSpec(nrows=5, ncols=2)
labels=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
times=['01_00','03_00','05_00','07_00','09_00']
sims=['CESM-ERA5-DJF-clim','CESM-2010-DJF-clim']
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'
q = 0
for k,sim in enumerate(sims):
    track = np.loadtxt(tracks + sim + '-new-tracks.txt')
    tt,lo,la,ID = track[:,0],track[:,1],track[:,2],track[:,-1]

    for l,t in enumerate(times):
        ax=fig.add_subplot(gs[l,k],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')

        ref=ds(wrfd + sim + '/wrfout_d01_2000-12-%s:00:00'%t)

        PVref=wrf.getvar(ref,'pvo')
        Pref = wrf.getvar(ref,'pressure')
        MSLPref = ref.variables['MSLP'][0,:]
        PV300ref = wrf.interplevel(PVref,Pref,300,meta=False)

        lon = wrf.getvar(ref,'lon')[0]
        lat = wrf.getvar(ref,'lat')[:,0]

        if t!='01_00':
            loc=np.where(ID==1)[0]
            if np.any(tt[loc]==(int(t[1])-1)*24):
                ax.plot(lo[loc],la[loc],color='slategrey',zorder=2)
                loco = np.where(tt[loc]==(int(t[1])-1)*24)[0][0]
                ax.scatter(lo[loc[loco]],la[loc[loco]],marker='*',color='purple',zorder=20)

            loc=np.where(ID==2)[0]
            if np.any(tt[loc]==(int(t[1])-1)*24):
                ax.plot(lo[loc],la[loc],color='slategrey',zorder=2)
                loco = np.where(tt[loc]==(int(t[1])-1)*24)[0][0]
                ax.scatter(lo[loc[loco]],la[loc[loco]],marker='*',color='purple',zorder=20)

        h2=ax.contour(lon,lat,MSLPref,levels=PSlevel,colors='purple',linewidths=0.5)
        hc=ax.contourf(lon,lat,PV300ref,cmap=pvcmap,norm=pvnorm,extend='both',levels=pvlevels)
        plt.clabel(h2,inline=True,fmt='%d',fontsize=6)
        ax.set_xlim(-90,45)
        ax.set_ylim(20,80)
        ax.set_yticks([25,40,55,70])
        ax.set_xticks([-80,-60,-40,-20,0,20,40])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        t=ax.text(0.02,0.9,'%s t = %s d'%(labels[q],str(int(t[:2])-1)),zorder=15,transform=ax.transAxes)
        t.set_bbox(dict(facecolor='white',edgecolor='white'))
        ax.set_aspect('auto')
        q+=1

plt.subplots_adjust(wspace=0.0,hspace=0)
pos0 = fig.get_axes()[0].get_position()
x0 = pos0.x0

ax = fig.get_axes()[-1]
pos = ax.get_position()

cbax = fig.add_axes([x0+pos.width/2, pos.y0-0.04, pos.width, 0.01])
cbar=plt.colorbar(hc, ticks=pvlevels,cax=cbax,orientation='horizontal')
pvticklabels[-1]='PVU'
cbar.ax.set_xticklabels(pvticklabels)
plt.subplots_adjust(wspace=0.0,hspace=0)
for ax in fig.get_axes()[:5]:
    ax.set_yticks([25,40,55,70])
    ax.set_yticklabels([r'25$^{\circ}$N',r'40$^{\circ}$N',r'55$^{\circ}$N',r'70$^{\circ}$N'])

for ax in fig.get_axes()[4::5]:
    ax.set_xticks([-80,-60,-40,-20,0,20,40])
    ax.set_xticklabels([r'80$^{\circ}$W',r'60$^{\circ}$W',r'40$^{\circ}$W',r'20$^{\circ}$W',r'0$^{\circ}$E',r'20$^{\circ}$E',r'40$^{\circ}$E'])

ax = fig.get_axes()[0]
ax.text(0.25,1.02,'ERA5 PD climatology', transform=ax.transAxes)

ax = fig.get_axes()[5]
ax.text(0.25,1.02,'CESM PD climatology',transform=ax.transAxes)

fig.savefig(pappath + 'ERA5-CESM-1981-2010-ref-comparison.png',dpi=300,bbox_inches='tight')
plt.close('all')


fig=plt.figure(figsize=(10,12))
gs = gridspec.GridSpec(nrows=5, ncols=2)
labels=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
times=['01_00','03_00','05_00','07_00','09_00']
sims=['CESM-2070-DJF-clim','CESM-2100-DJF-clim']
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'
q = 0
for k,sim in enumerate(sims):
    try:
        track = np.loadtxt(tracks + sim + '-new-tracks.txt')
        tt,lo,la,ID = track[:,0],track[:,1],track[:,2],track[:,-1]

    except:
        pass
    for l,t in enumerate(times):
        ax=fig.add_subplot(gs[l,k],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')

        ref=ds(wrfd + sim + '/wrfout_d01_2000-12-%s:00:00'%t)

        PVref=wrf.getvar(ref,'pvo')
        Pref = wrf.getvar(ref,'pressure')
        MSLPref = ref.variables['MSLP'][0,:]
        PV300ref = wrf.interplevel(PVref,Pref,300,meta=False)

        lon = wrf.getvar(ref,'lon')[0]
        lat = wrf.getvar(ref,'lat')[:,0]

        try:
         if t!='01_00':
            loc=np.where(ID==1)[0]
            if np.any(tt[loc]==(int(t[1])-1)*24):
                ax.plot(lo[loc],la[loc],color='slategrey',zorder=2)
                loco = np.where(tt[loc]==(int(t[1])-1)*24)[0][0]
                ax.scatter(lo[loc[loco]],la[loc[loco]],marker='*',color='purple',zorder=20)

            loc=np.where(ID==2)[0]
            if np.any(tt[loc]==(int(t[1])-1)*24):
                ax.plot(lo[loc],la[loc],color='slategrey',zorder=2)
                loco = np.where(tt[loc]==(int(t[1])-1)*24)[0][0]
                ax.scatter(lo[loc[loco]],la[loc[loco]],marker='*',color='purple',zorder=20)

        except:
            pass
        h2=ax.contour(lon,lat,MSLPref,levels=PSlevel,colors='purple',linewidths=0.5)
        hc=ax.contourf(lon,lat,PV300ref,cmap=pvcmap,norm=pvnorm,extend='both',levels=pvlevels)
        plt.clabel(h2,inline=True,fmt='%d',fontsize=6)
        ax.set_xlim(-90,45)
        ax.set_ylim(20,80)
        ax.set_yticks([25,40,55,70])
        ax.set_xticks([-80,-60,-40,-20,0,20,40])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        t=ax.text(0.02,0.9,'%s t = %s d'%(labels[q],str(int(t[:2])-1)),zorder=15,transform=ax.transAxes)
        t.set_bbox(dict(facecolor='white',edgecolor='white'))
        ax.set_aspect('auto')
        q+=1

plt.subplots_adjust(wspace=0.0,hspace=0)
pos0 = fig.get_axes()[0].get_position()
x0 = pos0.x0

ax = fig.get_axes()[-1]
pos = ax.get_position()

cbax = fig.add_axes([x0+pos.width/2, pos.y0-0.04, pos.width, 0.01])
cbar=plt.colorbar(hc, ticks=pvlevels,cax=cbax,orientation='horizontal')
pvticklabels[-1]='PVU'
cbar.ax.set_xticklabels(pvticklabels)
plt.subplots_adjust(wspace=0.0,hspace=0)
for ax in fig.get_axes()[:5]:
    ax.set_yticks([25,40,55,70])
    ax.set_yticklabels([r'25$^{\circ}$N',r'40$^{\circ}$N',r'55$^{\circ}$N',r'70$^{\circ}$N'])

for ax in fig.get_axes()[4::5]:
    ax.set_xticks([-80,-60,-40,-20,0,20,40])
    ax.set_xticklabels([r'80$^{\circ}$W',r'60$^{\circ}$W',r'40$^{\circ}$W',r'20$^{\circ}$W',r'0$^{\circ}$E',r'20$^{\circ}$E',r'40$^{\circ}$E'])

ax = fig.get_axes()[0]
ax.text(0.25,1.02,'CESM MC climatology', transform=ax.transAxes)

ax = fig.get_axes()[5]
ax.text(0.25,1.02,'CESM EC climatology',transform=ax.transAxes)

fig.savefig(pappath + 'CESM-2041-2070-2100-ref-comparison.png',dpi=300,bbox_inches='tight')
plt.close('all')
