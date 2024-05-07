import numpy as np
from netCDF4 import Dataset as ds
import wrf
import sys
sys.path.append('/home/ascherrmann/scripts/')
sys.path.append('/home/raphaelp/phd/scripts/basics/')
import helper
import matplotlib
import matplotlib.pyplot as plt
import cartopy
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import os
from colormaps import PV_cmap2
import argparse

parser = argparse.ArgumentParser(description="plot trajectory position evolution")
parser.add_argument('sim',default='',type=str,help='which reference state to perturb: mean or overlap')

args=parser.parse_args()
sim=str(args.sim)


cmap,levels,norm,ticklabels=PV_cmap2()

dwrf='/atmosdyn2/ascherrmann/013-WRF-sim/'

slplevels=np.arange(950,1031,5)

traj = np.loadtxt(dwrf + sim + '/trace.ll',skiprows=4)
rsh = 28+1
if 'hourly' in sim:
    rsh=84+1
t,lon,lat,z = traj[:,0].reshape(-1,rsh),traj[:,1].reshape(-1,rsh),traj[:,2].reshape(-1,rsh),traj[:,3].reshape(-1,rsh)
ran=np.arange(0,rsh,2)
date='20001204_12'
for q in ran[::2]:
    tt,lo,la,pre = t[0,q],lon[:,q],lat[:,q],z[:,0]-z[:,q]
    nd = helper.change_date_by_hours('%s'%date,tt)
    
    s = ds(dwrf + sim + '/wrfout_d01_%s-%s-%s:00:00'%(nd[:4],nd[4:6],nd[6:]),'r')
    Lon=wrf.getvar(s,'XLONG')[0]
    Lat=wrf.getvar(s,'XLAT')[:,0]
    pv=wrf.getvar(s,'pvo')
    p3d=wrf.getvar(s,'pressure')
    try:
        slp =s.variables['MSLP'][0]
    except:
        slp = wrf.getvar(s,'slp')
    s.close()
     
    pv300=wrf.interplevel(pv,p3d,300,meta=False)

    fig=plt.figure(figsize=(8,6))
    gs=gridspec.GridSpec(nrows=1, ncols=1)
    ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    cn=ax.contourf(Lon,Lat,pv300,cmap=cmap,norm=norm,levels=levels)
    cb=ax.contour(Lon,Lat,slp,levels=slplevels,colors='purple',linewidths=0.5)
    plt.clabel(cb,inline=True,fmt='%d',fontsize=6)
    ax.scatter(lo,la,marker='+',c=pre,s=2,cmap=matplotlib.cm.seismic,vmin=-500,vmax=500)
    
    pos = ax.get_position()
    ax.set_xlim(-150,40)
    ax.set_ylim(10,80)
    cax=fig.add_axes([pos.x0+pos.width,pos.y0,0.02,pos.height])
    cbar=plt.colorbar(cn,ticks=levels,cax=cax)
    cbar.ax.set_yticklabels(labels=np.append(ticklabels[:-1],'PVU'))
    fig.savefig(dwrf + sim + '/300-PV-%s.png'%nd,dpi=300,bbox_inches='tight')
    plt.close('all')
    


        
