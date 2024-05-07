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

fig=plt.figure(figsize=(12,6))
gs=gridspec.GridSpec(nrows=1, ncols=3)
for q,amp in enumerate(['0.7','1.4','2.1']):
 ax2=fig.add_subplot(gs[0,q])
 for col in lcol:
     ax2.plot([],[],color=col)

 for simid,sim in enumerate(SIMS):
    sea = sim[:3]
    if 'not' in sim or 'no-' in sim or '800' in sim:
       continue
    if se!=sea:
        continue
    if 'check' in sim:
        continue
    if sim[-4:]=='clim' or 'AT' in sim:
        continue

    if not amp in sim:
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

    locc = np.where(IDs==2)[0]
    ax2.plot(t[locc]/24,slp[locc],color=col,linestyle=ls)

    locc = np.where(IDs==1)[0]
    ax2.plot(t[locc]/24,slp[locc],color=col,linestyle=':')
    

axes=fig.get_axes()
for ax in axes:
    ax.set_xlabel('simulation time [d]')
    ax.set_ylim(960,1015)
    ax.set_xticks(ticks=np.arange(1,10,1))
    ax.set_xlim(0,9)

ax2=axes[1]
tex=ax2.text(0.04,0.93,'(b)',zorder=15,fontsize=8,transform=ax2.transAxes)
tex.set_bbox(dict(facecolor='white',edgecolor='white'))

ax=axes[0]
tex=ax.text(0.04,0.93,'(a)',zorder=15,fontsize=8,transform=ax.transAxes)
tex.set_bbox(dict(facecolor='white',edgecolor='white'))
ax.set_aspect('auto')

ax2.set_aspect('auto')
ax.set_ylabel('SLP [hPa]')

ax3=axes[2]
tex=ax3.text(0.04,0.93,'(c)',zorder=15,fontsize=8,transform=ax3.transAxes)
tex.set_bbox(dict(facecolor='white',edgecolor='white'))

plt.subplots_adjust(top=0.5)
ax.legend(ticklabels,fontsize=6,loc='lower left')
#pos=ax2.get_position()
#ax2.set_position([pos.x0+0.02,pos.y0,pos.width-0.015,pos.height])
fig.savefig('/home/ascherrmann/defense/DJF-Med-slps.png',dpi=300,bbox_inches='tight')
plt.close('all')

 
