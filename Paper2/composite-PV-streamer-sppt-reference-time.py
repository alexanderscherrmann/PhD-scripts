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

for se in seas:
  for amp in amps:
   for t in ['03_12','04_00','04_12','05_00','05_12','06_00','06_12','07_00','07_12']:
    avpv = np.zeros_like(ref.variables['T'][0,0])
    fig=plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    counter = 0
    
    for simid,sim in enumerate(SIMS):
        medid = np.array(MEDIDS[simid])
        atid = np.array(ATIDS[simid])
    
        sea = sim[:3]
        if se!=sea:
            continue

        if sim[-3]!='0':
            continue

        if float(sim[-12:-9])!=amp:
            continue

        data = ds(dwrf + sim + '/wrfout_d01_2000-12-%s:00:00'%t)
        pv = wrf.getvar(data,'pvo')
        p = wrf.getvar(data,'pressure')
        pv300 = wrf.interplevel(pv,p,300,meta=False)

        avpv+=pv300
        counter +=1
    hc = ax.contourf(LON[0],LAT[:,0],avpv/counter,levels=pvlevels,cmap=pvcmap,norm=pvnorm)

    ax.set_xlim(-20,40)
    ax.set_ylim(20,65)
    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(hc, ticks=pvlevels,cax=cbax)
    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.01)
    fig.canvas.mpl_connect('draw_event', func)

    fig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/sppt-med-streamer-%s-%s-%.1f.png'%(t,se,amp),dpi=300,bbox_inches='tight')
    plt.close('all')

