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
import wrfsims
SIMS,ATIDS,MEDIDS = wrfsims.sppt_ids()

from dypy.intergrid import Intergrid
from mpl_toolkits.basemap import Basemap
import helper

seasons = ['DJF','SON','SON']

pvpos = [[93,55],[95,56],[125,72]]
amps = [0.7,1.4,2.1,2.8]

pvcmap,pvlevels,pvnorm,pvticklabels=PV_cmap2()
ucmap,ulevels = cmocean.cm.tempo,np.arange(10,65,5)
unorm = BoundaryNorm(ulevels,ucmap.N)
PSlevel = np.arange(975,1031,5)
dis = 1500

dlat = helper.convert_radial_distance_to_lon_lat_dis_new(dis,0)

wrfd = '/atmosdyn2/ascherrmann/013-WRF-sim/'
pappath = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'
Fig = plt.figure(figsize=(12,15))
Gs = gridspec.GridSpec(nrows=3, ncols=3)
labels=[['(a)','(b)','(c)'],['(d)','(e)','(f)'],['(g)','(h)','(i)']]
amps = ['0.7','1.4','2.1']
days = ['02_00','04_00','06_00']

per = ds(wrfd + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00','r')
lon = wrf.getvar(per,'lon')[0]
lat = wrf.getvar(per,'lat')[:,0]

meanlvls = np.array([2,3,5,7])

stdmap = matplotlib.cm.terrain.reversed()
stdlvls = np.array([0,0.25,0.5,0.75,1,1.5,2,2.5,3,4])
stdnorm=BoundaryNorm(stdlvls,stdmap.N)

for q,da in enumerate(days):
    for l,am in enumerate(amps):
        counter = 0
        sims = np.array([])
        for sim in SIMS:
            if not 'SON' in sim or not am in sim:
                continue

            if 'check' in sim or 'not' in sim or 'nested' in sim:
                continue
            counter +=1
            sims=np.append(sims,sim)
            print(sim)

        ax = Fig.add_subplot(Gs[q,l],projection=ccrs.PlateCarree())

        std = np.zeros((counter,lat.size,lon.size))
        mean = np.zeros((lat.size,lon.size)) 
        for qw,sim in enumerate(sims):
            data = ds(wrfd + sim + '/wrfout_d01_2000-12-%s:00:00'%da,'r')
            pv = wrf.getvar(data,'pvo')
            pres = wrf.getvar(data,'pressure')
            pv300 = wrf.interplevel(pv,pres,300,meta=False)

            std[qw] = pv300
            mean+=pv300

        st = np.std(std,axis=0)
        mean/=counter
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
        cb=ax.contourf(lon,lat,st,levels=stdlvls,cmap=stdmap,norm=stdnorm)
        cn=ax.contour(lon,lat,mean,levels=meanlvls,colors='purple',linewidths=1)

        ax.set_aspect('auto')
        text = ax.text(0.03,0.9,labels[q][l],transform=ax.transAxes,zorder=100)
        text.set_bbox(dict(facecolor='white',edgecolor='white'))

        ax.set_xlim(-50,50)
        ax.set_ylim(25,80)

Fig.subplots_adjust(wspace=0,hspace=0,top=0.6)
ax0 = Fig.get_axes()[0]
pos0 = ax0.get_position()

pos = ax.get_position()
cbax = Fig.add_axes([pos.x0+pos.width,pos.y0,0.02,pos0.y0 + pos0.height - pos.y0])
cbar = plt.colorbar(cb, ticks=stdlvls,cax=cbax)


Fig.savefig(pappath + 'SON-PV-amplitude-time-std-sppt.png',dpi=300,bbox_inches='tight')
plt.close('all')
