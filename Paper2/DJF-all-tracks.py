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

SIMS,ATIDS,MEDIDS = wrfsims.sppt_ids()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]

cmap,levels,norm,ticklabels=PV_cmap2()

seas = np.array(['DJF','MAM','JJA','SON'])
amps = np.array([0.7,1.4,2.1])
dis = 1000

ymin = 100.
ymax = 1000.

deltas = 5.
mvcross    = Basemap()
pappath = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'
figg=plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax=figg.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)

fig,axx=plt.subplots(figsize=(8,6))

EWNS=['east','west','north','south']
marker=['p','d','s','+']
colors=['orange','green','navy','saddlebrown']
for simid,sim in enumerate(SIMS):
    if not 'DJF' in sim:
        continue
    mar='o'
    col = 'grey'
    for co,mark,ewns in zip(colors,marker,EWNS):
        if ewns in sim:
            mar = mark
            col=co

    print(sim,mar) 
    medid = np.array(MEDIDS[simid])
    sea = sim[:3]

    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]
    tra.close()

    for i in [1,2]:
        loc = np.where(IDs==i)[0]
        ax.plot(tlon[loc],tlat[loc],color=col,linewidth=0.2)
        loco = np.argmin(slp[loc])
        ax.scatter(tlon[loc[loco]],tlat[loc[loco]],color=col,marker='*',s=5,zorder=10)

    locc = np.where(IDs==1)[0]
    axx.scatter(slp[locc[np.argmin(slp[locc])]],slp[loc[loco]],color=col,marker=mar)
#    axx.text(slp[locc[np.argmin(slp[locc])]],slp[loc[loco]],sim[4:7]+sim[11:12])

    ax.set_xlim(-80,60)
    ax.set_ylim(25,80)
#    ax.set_extent([-120,80,20,80])
    figg.savefig(pappath + 'DJF-AT-MED-all-cyclone-tracks.png',dpi=300,bbox_inches='tight')
    plt.close(figg)
axx.set_xlabel('Atlantic cyclone intensity [hPa]')
axx.set_ylabel('Med cyclone intensity [hPa]')
fig.savefig(pappath + 'DJF-all-intensity-scatter.png',dpi=300,bbox_inches='tight')

plt.close('all')
