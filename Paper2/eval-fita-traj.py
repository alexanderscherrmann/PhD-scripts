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
cmap,levels,norm,ticklabels=PV_cmap2()
era5 = '/atmosdyn2/era5/cdf/'

Lat = np.arange(-90,90.1,0.5)
Lon = np.arange(-180,180,0.5)

region = [-8, 25, 28, 50]
slplevels=np.arange(950,1031,5)
#for date in ['10_00','11_00','11_12','12_00']:
for date in ['20040122_14','20191213_20','20191222_10']:
#    traj=np.loadtxt('/home/ascherrmann/fita-case-traj/fita-case-trajectories-200512%s.txt'%date,skiprows=4)
    p = era5 + '%s/%s/'%(date[:4],date[4:6])
    traj = np.loadtxt('/home/ascherrmann/fita-case-traj/fita-case-trajectories-%s.txt'%date,skiprows=4)

    t,lon,lat,pres = traj[:,0].reshape(-1,169),traj[:,1].reshape(-1,169),traj[:,2].reshape(-1,169),traj[:,3].reshape(-1,169)
    if not os.path.isdir('/home/ascherrmann/fita-case-traj/%s/'%date):
        os.mkdir('/home/ascherrmann/fita-case-traj/%s/'%date)
    ran=np.arange(0,169,6)
    for q in ran:
        tt,lo,la,pre = t[0,q],lon[:,q],lat[:,q],pres[:,q]-300

        nd = helper.change_date_by_hours('%s'%date,tt)
        
        sfile=p+'S%s'%nd
        mfile=p+'N%s'%nd
        M=ds(mfile,'r')
        s = ds(sfile,'r')
        PS = s.variables['PS'][0]
        pv = s.variables['PV'][0]
        slp = M.variables['MSL'][0]/100
        ak=s.variables['hyam'][137-98:]
        bk=s.variables['hybm'][137-98:]

        s.close()
        ps3d=np.tile(PS[:,:],(len(ak),1,1))
        p3d=(ak/100.+bk*ps3d.T).T
        
        pv300=wrf.interplevel(pv,p3d,300,meta=False)

        fig=plt.figure(figsize=(8,6))
        gs=gridspec.GridSpec(nrows=1, ncols=1)
        ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
        cn=ax.contourf(Lon,Lat,pv300,cmap=cmap,norm=norm,levels=levels)
        cb=ax.contour(Lon,Lat,slp,levels=slplevels,colors='purple',linewidths=0.5)
        plt.clabel(cb,inline=True,fmt='%d',fontsize=6)
        ax.scatter(lo,la,marker='+',c=pre,s=2,cmap=matplotlib.cm.seismic,vmin=-50,vmax=50)
        
        pos = ax.get_position()
        ax.set_xlim(-150,40)
        ax.set_ylim(10,80)
        cax=fig.add_axes([pos.x0+pos.width,pos.y0,0.02,pos.height])
        cbar=plt.colorbar(cn,ticks=levels,cax=cax)
        cbar.ax.set_yticklabels(labels=np.append(ticklabels[:-1],'PVU'))
        fig.savefig('/home/ascherrmann/fita-case-traj/%s/PV-%s.png'%(date,nd),dpi=300,bbox_inches='tight')
        plt.close('all')
        


        
