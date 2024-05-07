import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from netCDF4 import Dataset as ds
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
import wrf
import cartopy.crs as ccrs
import cartopy
from dypy.intergrid import Intergrid
from mpl_toolkits.basemap import Basemap
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
import helper
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import wrfsims
import matplotlib.gridspec as gridspec

SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')
LAT = wrf.getvar(ref,'lat')

cmap,levels,norm,ticklabels=PV_cmap2()

seas = np.array(['DJF','MAM','JJA','SON'])
amps = np.array([0.7,1.4,2.1])
dis = 1000

ymin = 100.
ymax = 1000.

deltas = 5.
mvcross    = Basemap()
pappath = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'



EWNS=['east','west','north','south']
marker=['p','d','s','+']
simc = ['200-km-west','400-km-west','200-km-east','400-km-east','200-km-north','400-km-north','200-km-south','400-km-south','clim-max']
ticklabels=['2W','4W','2E','4E','2N','4N','2S','4S','C']
lcol = ['grey','k','plum','purple','cyan','b','orange','r','saddlebrown']
cmap=matplotlib.cm.Reds
t='05_00'
clevels=np.arange(10,101,10)
se='DJF'
for amp in ['0.7','1.4','2.1']:
 fig=plt.figure(figsize=(11,5))
 gs=gridspec.GridSpec(nrows=1, ncols=3)
 ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())

 for col in lcol:
     ax.plot([],[],color=col)

 ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
 ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
 
 for simid,sim in enumerate(SIMS):
    sea = sim[:3]
    if 'not' in sim:
       continue
    if se!=sea:
        continue
    if 'check' in sim:
        continue
    if sim[-4:]=='clim' or 'AT' in sim:
        continue

    if float(sim[-8:-5])!=float(amp):
        continue
    ls='-'
    col = 'grey'

    for q,si in enumerate(simc):
        if si in sim:
            col=lcol[q]

    medid = np.array(MEDIDS[simid])
    sea = sim[:3]

    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]
 counter=0
 density = np.zeros_like(ref.variables['T'][0,0])
 for simid,sim in enumerate(SIMS):
     medid = np.array(MEDIDS[simid])
     atid = np.array(ATIDS[simid])

     sea = sim[:3]
     if 'not' in sim:
         continue
     if se!=sea:
         continue
     if 'check' in sim or 'AT' in sim:
         continue
     if sim[-4:]=='clim':
         continue
     if float(sim[-8:-5])!=float(amp):
         continue
     if 'AT' in sim:
         continue
     t='05_00'
     data = ds(dwrf + sim + '/wrfout_d01_2000-12-%s:00:00'%t)
     pv = wrf.getvar(data,'pvo')
     p = wrf.getvar(data,'pressure')
     pv300 = wrf.interplevel(pv,p,300,meta=False)
     for q,si in enumerate(simc):
         if si in sim:
             col=lcol[q]
     ax.contour(LON[0],LAT[:,0],pv300,levels=[2],colors=col,linewidths=0.5)

     density[pv300>=2]+=1
     counter +=1

 hb = ax.contourf(LON[0],LAT[:,0],density*100/counter,levels=clevels,cmap=cmap)

 ax.set_xlim(-45,45)
 ax.set_ylim(20,75)

 ax.set_xticks([-40,-20,0,20,40])
 ax.set_xticklabels(['40$^{\circ}$W','20$^{\circ}$W','0$^{\circ}$E','20$^{\circ}$E','40$^{\circ}$E'])
 ax.set_yticks([20,35,50,65])
 ax.set_yticklabels(['20$^{\circ}$N','35$^{\circ}$N','50$^{\circ}$N','65$^{\circ}$N'])

 ax.set_aspect('auto')

 pos=ax.get_position()

 plt.subplots_adjust(top=0.6)
 cbax = fig.add_axes([pos.x0+pos.width, pos.y0, 0.02, 0.6-pos.y0])
 cbar=plt.colorbar(hb, ticks=clevels,cax=cbax)
 cbar.ax.set_yticklabels(labels=np.append(clevels[:-1],r'%'))
 ax.legend(ticklabels,fontsize=6,loc='lower left')
 fig.savefig(pappath + 'DJF-%s-streamer-ridge.png'%amp,dpi=300,bbox_inches='tight')
 plt.close('all')

 
