import wrf
from netCDF4 import Dataset as ds
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
from colormaps import PV_cmap2
import wrfsims
import helper
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm

SIMS,ATIDS,MEDIDS = wrfsims.sppt_ids()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')
LAT = wrf.getvar(ref,'lat')

colors = ['dodgerblue','darkgreen','saddlebrown']
seas = np.array(['DJF','MAM','SON'])
amps = np.array([0.7,1.4,2.1])

pvcmap,pvlevels,pvnorm,ticklabels=PV_cmap2()
labels=['(a)','(b)','(c)']
levels=np.arange(0,3.1,0.3)
clevels=levels
cmap=matplotlib.cm.terrain
for se in seas[:1]:
  for t in ['03_00','03_12','04_00','04_12','05_00','05_12','06_00','06_12','07_00','07_12','08_00','08_12','09_00'][3:4]:
   gfig=plt.figure(figsize=(10,3))  
   ggs = gridspec.GridSpec(nrows=1, ncols=3)
   qq = 0 
   for amp in amps:

    fig=plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
    
    
    axx = gfig.add_subplot(ggs[0,qq],projection=ccrs.PlateCarree())
    axx.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    axx.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
    counter = 0
    density = np.zeros_like(ref.variables['T'][0,0])
    full3d=np.zeros((11,ref.variables['T'][0,0].shape[0],ref.variables['T'][0,0].shape[1]))
    qwp = 0 
    simc = np.append(np.array(['DJF-clim-max-U-at-300-hPa-%.1f-QGPV-%03d'%(amp,x)for x in range(1,11)]),'DJF-clim-max-U-at-300-hPa-%.1f-QGPV'%amp)
    print(simc)
    for simid,sim in enumerate(SIMS):
        if np.all(simc!=sim):
            continue
        print(sim)
        medid = np.array(MEDIDS[simid])
        atid = np.array(ATIDS[simid])

        sea = sim[:3]
    
        data = ds(dwrf + sim + '/wrfout_d01_2000-12-%s:00:00'%t)
        pv = wrf.getvar(data,'pvo')
        p = wrf.getvar(data,'pressure')
        pv300 = wrf.interplevel(pv,p,300,meta=False)
        density[pv300>=2]+=1
        full3d[qwp][pv300>=2]+=pv300[pv300>=2]
        qwp+=1

    
    vari = np.std(full3d,axis=0)
    vari[(density/qwp)<0.5]=np.nan
    hc = ax.contourf(LON[0],LAT[:,0],vari,levels=levels,cmap=cmap)
    
    ax.set_xlim(-20,40)
    ax.set_ylim(20,65)

    hb = axx.contourf(LON[0],LAT[:,0],vari,levels=levels,cmap=cmap)
    kk=4
    axx.set_xlim(-20,40)
    axx.set_ylim(20,65)
    axx.set_aspect('auto')
    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(hc, ticks=levels,cax=cbax)
    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.01)
    fig.canvas.mpl_connect('draw_event', func)
    cbar.ax.set_yticklabels(labels=np.append(np.round(levels[:-1],1),r'PVU'))

    fig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/composites-sppt-med-streamer/strength-vari-sppt-runs-med-streamer-%s-%s-%.1f.png'%(t,se,amp),dpi=300,bbox_inches='tight')
    plt.close(fig)
    tex=axx.text(-17.5,61.5,labels[qq],zorder=15,fontsize=8)
    tex.set_bbox(dict(facecolor='white',edgecolor='white'))
    qq+=1

   plt.subplots_adjust(hspace=0,wspace=0)
   pos=axx.get_position()
   cbax = gfig.add_axes([pos.x0+pos.width, pos.y0, 0.02, pos.height])
   cbar=plt.colorbar(hb, ticks=levels,cax=cbax)
   cbar.ax.set_yticklabels(labels=np.append(np.round(levels[:-1],2),r'PVU'))
   gfig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/composites-sppt-med-streamer/strength-vari-sppt-runs-med-streamer-%s-%s.png'%(t,se),dpi=300,bbox_inches='tight')
   plt.close(gfig)
