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
parser.add_argument('sim',default='',type=str,help='which reference state to perturb: mean or overlap')

args=parser.parse_args()
sim=str(args.sim)


cmap,levels,norm,ticklabels=PV_cmap2()

dwrf='/atmosdyn2/ascherrmann/013-WRF-sim/'

slplevels=np.arange(950,1031,5)

traj = np.loadtxt(dwrf + sim + '/wcb_trace.ll',skiprows=4)

with open(dwrf + sim + '/wcb_trace.ll') as f:
    fl = next(f)
f.close()

H = int(int(fl[-9:-5])/60/3)+1
rsh=H
t,lon,lat,pre = traj[:,0].reshape(-1,rsh),traj[:,1].reshape(-1,rsh),traj[:,2].reshape(-1,rsh),traj[:,-1].reshape(-1,rsh)
#dp=pre[:,0]-pre[:,-1]


if True:
    deli = np.array([])
    for q in range(lon[:,0].size):
        if np.any(lon[q]<=-100):
            deli=np.append(deli,q)
    deli=deli.astype(int)
    t = np.delete(t,deli,axis=0)
    lon=np.delete(lon,deli,axis=0)
    lat=np.delete(lat,deli,axis=0)
    pre=np.delete(pre,deli,axis=0)

if True:
    wcb = np.array([])
#    dp=pre-pre[:,0][:,None]
    for q in range(lon[:,0].size):
        if np.any(dp[q,:17]<=-500):
#        if np.any(pre[q,:17]<=400):
            wcb=np.append(wcb,q)
    wcb=wcb.astype(int)
    t,lon,lat,pre = t[wcb],lon[wcb],lat[wcb],pre[wcb]
#    wcb=np.where(dp>=150)[0]
#    t,lon,lat,pre = t[wcb],lon[wcb],lat[wcb],pre[wcb]


ran=np.arange(0,rsh,2)
date='20001201_03'

preslvl = np.arange(300,901,50)
presnorm=BoundaryNorm(preslvl,matplotlib.cm.jet.N)

for q in ran[::1]:
    tt,lo,la,pres = t[0,q],lon[:,q],lat[:,q],pre[:,q]
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

    lc2=ax.scatter(lo,la,marker='+',c=pres,s=2,cmap=matplotlib.cm.jet,norm=presnorm)
    
    ax.set_xlim(-120,40)
    ax.set_ylim(10,80)
    pos = ax.get_position()
    cax=fig.add_axes([pos.x0+pos.width,pos.y0,0.02,pos.height])
    cbar=plt.colorbar(cn,ticks=levels,cax=cax)
    cbar.ax.set_yticklabels(labels=np.append(ticklabels[:-1],'PVU'))
    cax=fig.add_axes([pos.x0,pos.y0-0.02,pos.width,0.02])
    cbar=plt.colorbar(lc2,ticks=np.arange(200,901,50),cax=cax,orientation='horizontal')
    cbar.ax.set_xticklabels(labels=np.arange(200,901,50))

    fig.savefig(dwrf + sim + '/wcb-300-PV-%s.png'%nd,dpi=300,bbox_inches='tight')
    plt.close('all')
    


        
