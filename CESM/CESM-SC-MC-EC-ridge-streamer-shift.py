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

SIMS,ATIDS,MEDIDS = wrfsims.cesm_ids()
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
pappath = '/atmosdyn2/ascherrmann/015-CESM-WRF/images/'

simc = ['200-km-west','400-km-west','200-km-east','400-km-east','200-km-north','400-km-north','200-km-south','400-km-south','-0-km']
ticklabels=['2W','4W','2E','4E','2N','4N','2S','4S','C']
lcol = ['grey','k','plum','purple','cyan','b','orange','r','saddlebrown']
cmap=matplotlib.cm.Reds
t='05_00'
clevels=np.arange(10,101,10)
se='DJF'

fig=plt.figure(figsize=(12,8))
gs=gridspec.GridSpec(nrows=1,ncols=3)
period = ['ERA5','2010','2040','2070','2100']
labels2= ['E2010 moderate','C2010 moderate','C2040 moderate','C2070 moderate','C2100 moderate']
amp='1.4'
labels=['(a)','(b)','(c)','(d)','(e)']
for q,perio in enumerate(period[2:]):
 ax = fig.add_subplot(gs[0,q],projection=ccrs.PlateCarree())
 for col in lcol:
     ax.plot([],[],color=col)
 
 ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
 ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)

 density = np.zeros_like(ref.variables['T'][0,0])
 counter=0
 for simid,sim in enumerate(SIMS):
    if not amp in sim or not perio in sim or '800' in sim:
        continue

    ls='-'
    col = 'grey'

    medid = np.array(MEDIDS[simid])
    atid = np.array(ATIDS[simid])

    t='05_00'
    data = ds(dwrf + sim + '/wrfout_d01_2000-12-%s:00:00'%t)
    pv = wrf.getvar(data,'pvo')
    p = wrf.getvar(data,'pressure')
    pv300 = wrf.interplevel(pv,p,300,meta=False)
    for qq,si in enumerate(simc):
        if si in sim:
            col=lcol[qq]
    ax.contour(LON[0],LAT[:,0],pv300,levels=[2],colors=col,linewidths=0.5)

    density[pv300>=2]+=1
    counter +=1

 hb = ax.contourf(LON[0],LAT[:,0],density*100/counter,levels=clevels,cmap=cmap)

 ax.set_xlim(-45,45)
 ax.set_ylim(20,75)

 if q==0:
  ax.set_xticks([-40,-20,0,20,40])
  ax.set_xticklabels(['40$^{\circ}$W','20$^{\circ}$W','0$^{\circ}$E','20$^{\circ}$E','40$^{\circ}$E'])
  ax0=ax
 else:
  ax.set_xticks([-20,0,20,40])
  ax.set_xticklabels(['20$^{\circ}$W','0$^{\circ}$E','20$^{\circ}$E','40$^{\circ}$E'])
 ax.set_aspect('auto')
 if q==0:
  ax.set_yticks([20,35,50,65])
  ax.set_yticklabels(['20$^{\circ}$N','35$^{\circ}$N','50$^{\circ}$N','65$^{\circ}$N'])

 ax.set_aspect('auto')
 if perio=='ERA5':
     ax.text(0.02,1.02,'%s %s'%(labels[q],labels2[q]),transform=ax.transAxes)
 else:
     ax.text(0.02,1.02,'%s %s'%(labels[q],labels2[q+2]),transform=ax.transAxes)



pos=ax.get_position()

plt.subplots_adjust(top=0.5,wspace=0,hspace=0)
cbax = fig.add_axes([pos.x0+pos.width, pos.y0, 0.02, 0.5-pos.y0])
cbar=plt.colorbar(hb, ticks=clevels,cax=cbax)
cbar.ax.set_yticklabels(labels=np.append(clevels[:-1],r'%'))
ax0.legend(ticklabels,fontsize=6,loc='lower left')
fig.savefig(pappath + 'CESM-SC-MC-EC-ridge-shift.png',dpi=300,bbox_inches='tight')
plt.close('all')

 
