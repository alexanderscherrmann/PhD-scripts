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
pappath = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

labels=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
times=['01_00','03_00','05_00','07_00','09_00']
sims=['DJF-clim','DJF-clim-max-U-at-300-hPa-1.4-QGPV','DJF-clim-max-U-at-300-hPa-0.7-QGPV','DJF-clim-max-U-at-300-hPa-2.1-QGPV']
for k,sim in enumerate(sims[:]):
    fig=plt.figure(figsize=(10,12))
    gs = gridspec.GridSpec(nrows=5, ncols=1)
    pos=[]
    q=0
    track = np.loadtxt(tracks + sim + '-new-tracks.txt')
    tt,lo,la,ID = track[:,0],track[:,1],track[:,2],track[:,-1]

    for l,t in enumerate(times):

        ax=fig.add_subplot(gs[l,0],projection=ccrs.PlateCarree())
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
        plt.clabel(h2,inline=True,fmt='%d',fontsize=4)
        
        #if k==0 and l==3:
        #    pos =ax.get_position()
        #    x0 = pos.x0

        ax.set_xlim(-90,45)
        ax.set_ylim(20,80)
        pos.append(ax.get_position())

        t=ax.text(0.025,0.86,labels[q],zorder=15,transform=ax.transAxes)
        t.set_bbox(dict(facecolor='white',edgecolor='white'))
        #ax.set_aspect('auto')
        q+=1

    pvticklabels[-1]='PVU'
#    pvticklabels[-2]=''
    plt.subplots_adjust(wspace=0.0,hspace=0)
    po5 = plt.gcf().get_axes()[-1].get_position()
    cbax = fig.add_axes([po5.x0, po5.y0-0.01, po5.width, 0.01])
    cbar=plt.colorbar(hc, ticks=pvlevels,cax=cbax,orientation='horizontal')
    cbar.ax.set_xticklabels(pvticklabels)
    

    fig.savefig(pappath + '%s-indv-dynamics.png'%sim,dpi=300,bbox_inches='tight')
    plt.close('all')
