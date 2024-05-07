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
lcol = ['cyan','navy','orange','red','grey','k','plum','purple','saddlebrown']
cmap=matplotlib.cm.Reds
t='04_12'
clevels=np.arange(10,101,10)
se='MAM'
for amp in ['0.7']:#,'1.4','2.1']:
 fig=plt.figure(figsize=(12,6))
 gs=gridspec.GridSpec(nrows=1, ncols=3)
 ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
 ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
 ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
 
 ax2=fig.add_subplot(gs[0,1])
 ax3 = fig.add_subplot(gs[0,2],projection=ccrs.PlateCarree())
 ax3.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
 ax3.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)

 for col in lcol:
     ax2.plot([],[],color=col)
     ax3.plot([],[],color=col)
 for simid,sim in enumerate(SIMS):
    sea = sim[:3]
    if 'not' in sim:
       continue
    if se!=sea:
        continue
    if 'check' in sim:
        continue
#    if sim[-4:]=='clim' or 'AT' in sim:
#        continue

#    if float(sim[-8:-5])!=float(amp):
    if not amp in sim and sim!='MAM-clim':
         continue
    print(sim)
    print(MEDIDS[simid])
    print(ATIDS[simid])
    ls='-'
    col = 'grey'

    for q,si in enumerate(simc):
        if si in sim:
            col=lcol[q]

    if sim=='MAM-clim':
        col='dodgerblue'
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

    locc = np.where(IDs==2)[0]
    ax2.plot(t[locc]/24,slp[locc],color=col,linestyle=ls)

 tex=ax2.text(0.04,0.93,'(b)',zorder=15,fontsize=8,transform=ax2.transAxes)
 tex.set_bbox(dict(facecolor='white',edgecolor='white'))
 tex=ax.text(0.04,0.93,'(a)',zorder=15,fontsize=8,transform=ax.transAxes)
 tex.set_bbox(dict(facecolor='white',edgecolor='white'))
 ax.set_aspect('auto')
 ax2.set_aspect('auto')
 ax2.set_xlabel('simulation time [d]')
 ax2.set_ylabel('SLP [hPa]')

 counter = 0
 ref = ds(dwrf + 'MAM-clim/wrfout_d01_2000-12-01_00:00:00')
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
#     if sim[-4:]=='clim':
#         continue
#     if float(sim[-8:-5])!=float(amp):
     if not amp in sim and sim!='MAM-clim':         
         continue
     if 'AT' in sim:
         continue
     print(sim)
     t='04_12'
     data = ds(dwrf + sim + '/wrfout_d01_2000-12-%s:00:00'%t)
     pv = wrf.getvar(data,'pvo')
     p = wrf.getvar(data,'pressure')
     pv300 = wrf.interplevel(pv,p,300,meta=False)
     for q,si in enumerate(simc):
         if si in sim:
             col=lcol[q]
     if sim=='MAM-clim':
         col='dodgerblue'

     ax3.contour(LON[0],LAT[:,0],pv300,levels=[2],colors=col,linewidths=0.5)

     density[pv300>=2]+=1
     counter +=1

 hb = ax3.contourf(LON[0],LAT[:,0],density*100/counter,levels=clevels,cmap=cmap)

 ax3.set_xlim(-20,40)
 ax3.set_ylim(20,65)
 ax3.set_aspect('auto')

 tex=ax3.text(0.04,0.93,'(c)',zorder=15,fontsize=8,transform=ax3.transAxes)
 tex.set_bbox(dict(facecolor='white',edgecolor='white'))

 pos=ax3.get_position()

 plt.subplots_adjust(top=0.5)
 cbax = fig.add_axes([pos.x0+pos.width, pos.y0, 0.02, 0.5-pos.y0])
 cbar=plt.colorbar(hb, ticks=clevels,cax=cbax)
 cbar.ax.set_yticklabels(labels=np.append(clevels[:-1],r'%'))
 ax.set_xlim(-80,60)
 ax.set_ylim(25,80)
 ax2.set_xticks(ticks=np.arange(3,10,1))
 ax2.legend(ticklabels,fontsize=6,loc='upper right')
 pos=ax2.get_position()
 ax2.set_position([pos.x0+0.02,pos.y0,pos.width,pos.height])
 fig.savefig(pappath + 'MAM-%s-tracks-Med-slp.png'%amp,dpi=300,bbox_inches='tight')
 plt.close('all')

 
