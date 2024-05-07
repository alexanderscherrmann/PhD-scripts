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
pappath = '/atmosdyn2/ascherrmann/015-CESM-WRF/'

EWNS=['east','west','north','south']
marker=['p','d','s','+']
simc = ['200-km-west','400-km-west','200-km-east','400-km-east','200-km-north','400-km-north','200-km-south','400-km-south','-0-km-']
ticklabels=['2W','4W','2E','4E','2N','4N','2S','4S','C']
lcol = ['grey','k','plum','purple','cyan','b','orange','r','saddlebrown']
cmap=matplotlib.cm.Reds
t='05_00'
clevels=np.arange(10,101,10)
se='DJF'


names = ['-0-km','west','east','south','north']
km=['-0-km','200','400','800']
period=['ERA5','2010','2040','2070','2100']

for perio in period:
    if perio!='ERA5':
        continue

    for amp in ['0.7','1.4','2.1']:
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
        ls='-'
        col = 'grey'
        if not perio in sim or '800' in sim or not amp in sim:
            continue
        for q,si in enumerate(simc):
            if si in sim:
                col=lcol[q]
    
        medid = np.array(MEDIDS[simid])
        if np.any(medid==None):
            continue
        
        tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
        t = tra[:,0]
        tlon,tlat = tra[:,1],tra[:,2]
        slp = tra[:,3]
        IDs = tra[:,-1]
    
        for i in [1,2]:
            loc = np.where(IDs==i)[0]
            loco = np.argmin(slp[loc])
            if i==1:
             ax.plot(tlon[loc],tlat[loc],color=col,linestyle=ls)
             ax.scatter(tlon[loc[loco]],tlat[loc[loco]],color=col,marker='*',s=50,zorder=10)
            else:
                ax3.plot(tlon[loc],tlat[loc],color=col,linestyle=ls)
                ax3.scatter(tlon[loc[loco]],tlat[loc[loco]],color=col,marker='*',s=50,zorder=10)
    
        locc = np.where(IDs==2)[0]
        ax2.plot(t[locc]/24,slp[locc],color=col,linestyle=ls)
    
        locc = np.where(IDs==1)[0]
        ax2.plot(t[locc]/24,slp[locc],color=col,linestyle=':')
    
     tex=ax2.text(0.04,0.93,'(b)',zorder=15,fontsize=8,transform=ax2.transAxes)
     tex.set_bbox(dict(facecolor='white',edgecolor='white'))
     tex=ax.text(0.04,0.93,'(a)',zorder=15,fontsize=8,transform=ax.transAxes)
     tex.set_bbox(dict(facecolor='white',edgecolor='white'))
     ax.set_aspect('auto')
    
     ax2.set_aspect('auto')
     ax2.set_xlabel('simulation time [d]')
     ax2.set_ylabel('SLP [hPa]')
    
     ax3.set_xlim(-10,45)
     ax3.set_ylim(20,55)
     ax3.set_xticks([-10,-0,10,20,30,40])
     ax3.set_xticklabels(['10$^{\circ}$W','0$^{\circ}$E','10$^{\circ}$E','20$^{\circ}$E','30$^{\circ}$E','40$^{\circ}$E'])
     ax3.set_yticks([20,30,40,50])
     ax3.set_yticklabels(['20$^{\circ}$N','30$^{\circ}$N','40$^{\circ}$N','50$^{\circ}$N'])
    
     ax3.set_aspect('auto')
    
     tex=ax3.text(0.04,0.93,'(c)',zorder=15,fontsize=8,transform=ax3.transAxes)
     tex.set_bbox(dict(facecolor='white',edgecolor='white'))
    
     pos=ax3.get_position()
    
     plt.subplots_adjust(top=0.5)
     ax.set_xlim(-70,20)
     ax.set_xticks([-60,-40,-20,0,20,40])
     ax.set_xticklabels(['60$^{\circ}$W','40$^{\circ}$W','20$^{\circ}$W','0$^{\circ}$E','20$^{\circ}$E','40$^{\circ}$E'])
     ax.set_yticks([40,60,80])
     ax.set_yticklabels(['40$^{\circ}$N','60$^{\circ}$N','80$^{\circ}$N'])
     ax.set_ylim(35,80)
     ax2.set_xticks(ticks=np.arange(1,10,1))
     ax2.legend(ticklabels,fontsize=6,loc='lower left')
     pos=ax2.get_position()
     ax2.set_position([pos.x0+0.02,pos.y0,pos.width-0.015,pos.height])
     fig.savefig(pappath + 'CESM-%s-%s-tracks-Med-slp.png'%(perio,amp),dpi=300,bbox_inches='tight')
     plt.close('all')

 
