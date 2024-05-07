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
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]

cmap,levels,norm,ticklabels=PV_cmap2()

dis = 1000

ymin = 100.
ymax = 1000.

deltas = 5.
mvcross    = Basemap()
pappath = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'

colors=['grey','cyan','navy','purple','yellow','orange','lightcoral','red']
amps=['%.1f'%x for x in [0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 1.7, 2.1]]
print(amps)

fig=plt.figure(figsize=(10,6))
gs=gridspec.GridSpec(nrows=1, ncols=3)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
ax2=fig.add_subplot(gs[0,1])
ax3=fig.add_subplot(gs[0,2])

for amp,col in zip(amps,colors):
 for simid,sim in enumerate(SIMS):
    if not amp in sim:
        continue
    if sim[-3]=='0':
        continue
    if not 'DJF' in sim:
        continue
    if not 'clim' in sim:
        continue
    ls='-'
    mark='o'
    if 'check' in sim:
        ls=':'
        mark='s'
    print(sim,mark) 
    medid = np.array(MEDIDS[simid])
    sea = sim[:3]

    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]

    for i in [1,2]:
        loc = np.where(IDs==i)[0]
        ax.plot(tlon[loc],tlat[loc],color=col,linestyle=ls)
        loco = np.argmin(slp[loc])
        ax.scatter(tlon[loc[loco]],tlat[loc[loco]],color=col,marker='*',s=50,zorder=10)
        if i==1:
            aminslp=np.min(slp[loc])
        else:
            minslp=np.min(slp[loc])


    locc = np.where(IDs==2)[0]
    ax2.plot(t[locc],slp[locc],color=col,linestyle=ls)
    ax3.scatter(aminslp,minslp,color=col,marker=mark)

ax.set_aspect('auto')
ax2.set_aspect('auto')
ax2.set_xlabel('simulation time [h]')
ax2.set_ylabel('SLP [hPa]')
ax3.set_aspect('auto')
plt.subplots_adjust(top=0.5)
ax.set_xlim(-80,60)
ax.set_ylim(25,80)
ax2.set_xticks(ticks=np.arange(72,217,24))
fig.savefig(pappath + 'DJF-comparison-tracks-Med-slp.png',dpi=300,bbox_inches='tight')
plt.close('all')
