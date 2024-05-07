import wrf
from netCDF4 import Dataset as ds
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec
import matplotlib
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert


pwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
osim = 'DJF-clim-max-U-at-300-hPa-1.4-QGPV/'
nsim = 'nested-test/'

#tcomp = ['04_00','05_00','06_00','07_00','08_00','08_06','08_12','08_18']

wrfout = 'wrfout_d01_2000-12-'
end = ':00:00'

minlon = -100
maxlon = 80
minlat = 10
maxlat = 80

cmap = matplotlib.cm.BrBG
levels = np.arange(-2,2.1,0.4)

for d in range(1,10):
  for ho in np.arange(0,19,6):
    t = '%02d_%02d'%(d,ho)
    data = ds(pwrf + osim + wrfout + t + end,'r')
    pv = wrf.getvar(data,'pvo')
    pres = wrf.getvar(data,'pressure')

    oldPV = wrf.interplevel(pv,pres,300,meta=False)

    data = ds(pwrf + nsim + wrfout + t + end,'r')
    pv = wrf.getvar(data,'pvo')
    pres = wrf.getvar(data,'pressure')

    newPV = wrf.interplevel(pv,pres,300,meta=False)
    LON = wrf.getvar(data,'lon')[0]
    LAT = wrf.getvar(data,'lat')[:,0]
    dPV = newPV-oldPV


    fig = plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=2, edgecolor='black')
    
    h = ax.contourf(LON,LAT,dPV,cmap=cmap,levels=levels,extend='both')
    ax.set_xlim(minlon,maxlon)
    ax.set_ylim(minlat,maxlat)
    ax.set_extent([minlon,maxlon,minlat,maxlat])
    cbax = fig.add_axes([0.0, 0.0, 0.1, 0.1])
    bar=plt.colorbar(h, ticks=levels,cax=cbax)
    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.015)
    fig.canvas.mpl_connect('draw_event', func)

    fig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/upper-level-delta-PV-at-%s.png'%t,dpi=300,bbox_inches='tight')
    plt.close('all')
    



    


