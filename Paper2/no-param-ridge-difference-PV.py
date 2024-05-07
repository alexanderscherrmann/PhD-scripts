import numpy as np
from netCDF4 import Dataset as ds
import wrf
import sys
sys.path.append('/home/ascherrmann/scripts/')
sys.path.append('/home/raphaelp/phd/scripts/basics/')
import helper
import matplotlib
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import cartopy
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import os
from colormaps import PV_cmap2
import argparse

parser = argparse.ArgumentParser(description="plot trajectory position evolution")
parser.add_argument('sim1',default='',type=str,help='which reference state to perturb: mean or overlap')
parser.add_argument('sim2',default='',type=str,help='which reference state to perturb: mean or overlap')

args=parser.parse_args()
sim1=str(args.sim1)
sim2=str(args.sim2)

cmap,levels,norm,ticklabels=PV_cmap2()
cmap=matplotlib.cm.coolwarm

dwrf='/atmosdyn2/ascherrmann/013-WRF-sim/'

date='20001201_00'

dpvlvl = np.arange(-3,3.1,0.5)
dpvnorm=BoundaryNorm(dpvlvl,cmap.N)

levels=dpvlvl

for tt in range(6,73,6):
    nd = helper.change_date_by_hours('%s'%date,tt)
    
    s = ds(dwrf + sim1 + '/wrfout_d01_%s-%s-%s:00:00'%(nd[:4],nd[4:6],nd[6:]),'r')
    Lon=wrf.getvar(s,'XLONG')[0]
    Lat=wrf.getvar(s,'XLAT')[:,0]
    pv=wrf.getvar(s,'pvo')
    p3d=wrf.getvar(s,'pressure')
    s.close()
     
    pv300_1=wrf.interplevel(pv,p3d,300,meta=False)

    s = ds(dwrf + sim2 + '/wrfout_d01_%s-%s-%s:00:00'%(nd[:4],nd[4:6],nd[6:]),'r')
    pv=wrf.getvar(s,'pvo')
    p3d=wrf.getvar(s,'pressure')
    s.close()

    pv300_2=wrf.interplevel(pv,p3d,300,meta=False)

    fig=plt.figure(figsize=(8,6))
    gs=gridspec.GridSpec(nrows=1, ncols=1)
    ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    cn=ax.contourf(Lon,Lat,pv300_1-pv300_2,cmap=cmap,norm=dpvnorm,levels=levels,extend='both')

    
    ax.set_xlim(-120,40)
    ax.set_ylim(10,80)

    pos = ax.get_position()
    cax=fig.add_axes([pos.x0+pos.width,pos.y0,0.02,pos.height])
    cbar=plt.colorbar(cn,ticks=levels,cax=cax)
    cbar.ax.set_yticklabels(labels=np.append(levels[:-1],'PVU'))

    fig.savefig(dwrf + sim2 + '/delta-PV-no-param-%s.png'%nd,dpi=300,bbox_inches='tight')
    plt.close('all')
    


        
