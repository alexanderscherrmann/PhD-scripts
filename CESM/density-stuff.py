import pickle
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


dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]


denslvl = np.arange(10,101,10)
#denslvl = np.arange(10,71,10)
cmap = matplotlib.cm.Reds
norm=BoundaryNorm(denslvl,cmap.N)

flier = dict(marker='+',markerfacecolor='grey',markersize=1,linestyle=' ',markeredgecolor='grey')
meanline = dict(linestyle='-',linewidth=1,color='red')
meanline2 = dict(linestyle='-',linewidth=1,color='dodgerblue')
capprops = dict(linestyle='-',linewidth=1,color='grey')
medianprops = dict(linestyle='-',linewidth=1,color='black')
boxprops = dict(linestyle='-',linewidth=1.,color='slategrey')
whiskerprops= dict(linestyle='-',linewidth=1,color='grey')
labels=['(a)','(b)','(c)']

lab = np.arange(3,217,3)

for sea in ['DJF','MAM','SON']:
    if sea!='DJF':
        continue

    f = open(dwrf + 'data/%s-density-data-slp.txt'%sea,'rb')
    d = pickle.load(f)
    f.close()
    
    atc = d['atcount']
    atdens = d['atdens']
    atslp = d['atslp']
    attrd = d['attrackdens']
    
    medc = d['medcount']
    meddens = d['meddens']
    medslp = d['medslp']
    medtrd = d['medtrackdens']
    
    dens = dict()
    figg=plt.figure(figsize=(10,8))
    ggs=gridspec.GridSpec(nrows=1, ncols=3)
    for qqq,am in enumerate(['0.7','1.4','2.1']):

        fig = plt.figure(figsize=(10,6))
        gs=gridspec.GridSpec(nrows=2, ncols=2)
    
        count = 0
        trackdens = np.zeros_like(atdens[am])
        alltrackdens = np.zeros_like(trackdens)
        simcount = 0
        for k in atc[am].keys():
            if am in k:
                count+=atc[am][k]
                trackdens+=attrd[am][k]
                if am=='0.7' and np.any(medtrd[am][k][35:64,240:261]!=0):
                    medtrd[am][k][35:64,230:261]=0
                alltrackdens+=attrd[am][k]
                alltrackdens+=medtrd[am][k]
                simcount+=1
    
        trackdens/=simcount
        alltrackdens/=simcount

        meanbox = []
        for tt in atslp[am].keys():
            meanbox.append(np.sort(atslp[am][tt]))
    
        dens[am] = atdens[am]/count
    
        ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
        if denslvl[-1]==100:
            cf = ax.contourf(LON,LAT,trackdens*100,cmap=cmap,levels=denslvl,norm=norm)
        else:
            cf = ax.contourf(LON,LAT,dens[am]*100,cmap=cmap,levels=denslvl,norm=norm)
    
        ax.set_xlim(-70,30)
        ax.set_ylim(30,80)
        ax.set_aspect('auto')
    
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0,pos.y0-0.02,pos.width,0.02])
        cbar = plt.colorbar(cf,cax=cax,orientation='horizontal')
    #    cbar.ax.set_xticklabels(np.arange(10,80,10))
        cbar.ax.set_xticklabels(denslvl)
    
        ax=fig.add_subplot(gs[0,1]) 
        bp = ax.boxplot(meanbox,whis=(10,90),labels=lab,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=True,medianprops=medianprops)
        ax.set_ylabel('SLP [hPa]')
        ax.set_xlim(12,72)
        ax.set_ylim(960,1010)
        ax.set_xticks(ticks=np.arange(4,73,4))
        ax.set_xticklabels(labels=np.ravel([np.array(['' for x in np.arange(1,10,1)]),(np.arange(1,19,1)[1::2]/2).astype(int)],'F'))
        ax.set_xlabel('simulation time [d]')
    
        ax.set_aspect('auto')
    
        count = 0
        trackdens = np.zeros_like(atdens[am])
        simcount = 0
        for k in atc[am].keys():
            if am in k:
                count+=atc[am][k]
                trackdens+=medtrd[am][k]
                simcount+=1
    
        trackdens/=simcount
    
        meanbox = []
        for tt in medslp[am].keys():
            meanbox.append(np.sort(medslp[am][tt]))
    
        dens[am] = meddens[am]/count
    
        ax=fig.add_subplot(gs[1,0],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
        if denslvl[-1]==100:
            ax.contourf(LON,LAT,trackdens*100,cmap=cmap,levels=denslvl,norm=norm)
    
        else:
            ax.contourf(LON,LAT,dens[am]*100,cmap=cmap,levels=denslvl,norm=norm)
    
        ax.set_xlim(-10,50)
        ax.set_ylim(25,50)
        ax.set_aspect('auto')
    
        ax=fig.add_subplot(gs[1,1])
        bp = ax.boxplot(meanbox,whis=(10,90),labels=lab,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=True,medianprops=medianprops)
        ax.set_ylabel('SLP [hPa]')
        ax.set_xlim(0,72)
        ax.set_ylim(990,1015)
        ax.set_xticks(ticks=np.arange(4,73,4))
        ax.set_xticklabels(labels=np.ravel([np.array(['' for x in np.arange(1,10,1)]),(np.arange(1,19,1)[1::2]/2).astype(int)],'F'))
    #    ax.set_xticklabels(labels=np.arange(12,217,12)/24)
        ax.set_xlabel('simulation time [d]')
        ax.set_aspect('auto')
        fig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/%s-track-density-box-slp-%s.png'%(sea,am),dpi=300,bbox_inches="tight")
        plt.close(fig)

        ax=figg.add_subplot(ggs[0,qqq],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
        cf = ax.contourf(LON,LAT,alltrackdens*100,cmap=cmap,levels=denslvl,norm=norm)
        tex = ax.text(0.03,0.9,labels[qqq],transform=ax.transAxes)
        tex.set_bbox(dict(edgecolor='white',facecolor='white'))
        ax.set_xlim(-70,45)
        ax.set_ylim(25,75)
    figg.subplots_adjust(wspace=0,hspace=0)

    ax = figg.get_axes()[-1]
    pos = ax.get_position()
    cax = figg.add_axes([pos.x0+pos.width,pos.y0,0.02,pos.height])
    cbar = plt.colorbar(cf,cax=cax)
    cbar.ax.set_yticklabels(np.append(denslvl[:-1],'%'))
    figg.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/%s-alltrackdens.png'%sea,dpi=300,bbox_inches="tight")
    plt.close(figg)







