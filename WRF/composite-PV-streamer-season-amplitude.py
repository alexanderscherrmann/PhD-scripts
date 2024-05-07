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

SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/home/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')
LAT = wrf.getvar(ref,'lat')

track20 = 0


colors = ['dodgerblue','darkgreen','r','saddlebrown']
seas = np.array(['DJF','MAM','JJA','SON'])
pos = np.array(['east','north','south','west'])
amps = np.array([0.7,1.4,2.1])

pvcmap,pvlevels,pvnorm,ticklabels=PV_cmap2()

for se in seas:
  for amp in amps:

    avpv = np.zeros_like(ref.variables['T'][0,0])
    fig=plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    counter = 0

    for simid2,sim in enumerate(SIMS):
        simid = simid2+track20
        if "-not" in sim:
            continue
    
        if sim[-4:]=='clim':
            continue
        medid = np.array(MEDIDS[simid])
        atid = np.array(ATIDS[simid])
    
        if medid.size==0:
            continue
    
        sea = sim[:3]
        if se!=sea:
            continue
        if float(sim[-8:-5])!=amp:
            continue

        tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
        t = tra[:,0]
        tlon,tlat = tra[:,1],tra[:,2]
        slp = tra[:,3]
        IDs = tra[:,-1]
    
        loc = np.where(IDs==2)[0] 
        t0 = t[loc[0]]
        dt = helper.simulation_time_to_day_string(t0)
        data = ds(dwrf + sim + '/wrfout_d01_2000-12-%s:00:00'%dt)
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

    fig.savefig(dwrf + 'image-output/med-streamer-%s-%.1f.png'%(se,amp),dpi=300,bbox_inches='tight')
    plt.close('all')
