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
labels=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
strength=['weak','moderate','strong']

figg=plt.figure(figsize=(10,8))
fig=plt.figure(figsize=(10,8))
fi=plt.figure(figsize=(10,8))

gs=gridspec.GridSpec(nrows=3, ncols=3)


lab = np.arange(3,217,3)

for Sea,(row,sea) in zip(['Winter','Spring','Autumn'],enumerate(['DJF','MAM','SON'])):

    f = open(dwrf + 'data/%s-density-data-slp-no-sppt.txt'%sea,'rb')
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
    for col,am in enumerate(['0.7','1.4','2.1']):

        count = 0
        alltrackdens = np.zeros_like(atdens[am])
        simcount = 0
        for k in atc[am].keys():
            if am in k:
                count+=atc[am][k]
                if sea=='DJF' and am=='0.7' and np.any(medtrd[am][k][35:64,240:261]!=0):
                    medtrd[am][k][35:64,230:261]=0
                alltrackdens+=attrd[am][k]
                alltrackdens+=medtrd[am][k]
                simcount+=1
    
        alltrackdens/=simcount

        meanbox = []
        for tt in atslp[am].keys():
            meanbox.append(np.sort(atslp[am][tt]))
    
        ax=figg.add_subplot(gs[row,col],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
        alltrackdens[alltrackdens>1]=1
        cf = ax.contourf(LON,LAT,alltrackdens*100,cmap=cmap,levels=denslvl,norm=norm)
        tex = ax.text(0.03,0.05,'%s %s %s'%(labels[3*row+col],Sea,strength[col]),transform=ax.transAxes,zorder=1000)
        tex.set_bbox(dict(edgecolor='white',facecolor='white'))
        ax.set_xlim(-70,45)
        ax.set_ylim(25,75)
        ax.set_aspect('auto')
    
        axx=fig.add_subplot(gs[row,col])

        bp = axx.boxplot(meanbox,whis=(10,90),labels=lab,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=True,medianprops=medianprops)
        if col==0:
            axx.set_ylabel('SLP [hPa]')
#            if row!=2:
#                axx.set_yticklabels(np.append('',np.arange(970,1016,10)))

        else:
            axx.set_yticklabels([])

        axx.set_xlim(12,72)
        axx.set_ylim(960,1015)
        axx.set_xticks(ticks=np.arange(4,73,4))
        if row==2:

            axx.set_xticklabels(labels=np.ravel([np.array(['' for x in np.arange(1,10,1)]),(np.arange(1,19,1)[1::2]/2).astype(int)],'F'))
            axx.set_xlabel('simulation time [d]')
        else:
            axx.set_xticklabels([])
        tex = axx.text(0.03,0.9,'%s %s %s'%(labels[3*row+col],Sea,strength[col]),transform=axx.transAxes)
        tex.set_bbox(dict(edgecolor='white',facecolor='white'))
 
        axx.set_aspect('auto')
    
        meanbox = []
        for tt in medslp[am].keys():
            meanbox.append(np.sort(medslp[am][tt]))
    
        axxx=fi.add_subplot(gs[row,col])
        bp = axxx.boxplot(meanbox,whis=(10,90),labels=lab,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=True,medianprops=medianprops)
        if col==0:
            axxx.set_ylabel('SLP [hPa]')
            if row!=2:
                axxx.set_yticklabels(np.append('',np.arange(995,1016,5)))
        else:
            axxx.set_yticklabels([])


        if col!=0:
            axxx.set_yticklabels([])
        axxx.set_xlim(0,72)

        axxx.set_ylim(990,1015)
        axxx.set_xticks(ticks=np.arange(4,73,4))
        if row==2:
            axxx.set_xticklabels(labels=np.ravel([np.array(['' for x in np.arange(1,10,1)]),(np.arange(1,19,1)[1::2]/2).astype(int)],'F'))
            axxx.set_xlabel('simulation time [d]')
        else:
            axxx.set_xticklabels([])
        tex = axxx.text(0.03,0.05,'%s %s %s'%(labels[3*row+col],Sea,strength[col]),transform=axxx.transAxes)
        tex.set_bbox(dict(edgecolor='white',facecolor='white'))

        axxx.set_aspect('auto')

figg.subplots_adjust(wspace=0,hspace=0)
fig.subplots_adjust(wspace=0,hspace=0)
fi.subplots_adjust(wspace=0,hspace=0)

#ax.set_xlim(-70,45)
#ax.set_ylim(25,75

for ax in figg.get_axes()[::3]:
    ax.set_yticks([25,40,55,70])
    ax.set_yticklabels([r'25$^{\circ}$N',r'40$^{\circ}$N',r'55$^{\circ}$N',r'70$^{\circ}$N'])

for ax in figg.get_axes()[-3:]:
    ax.set_xticks([-70,-50,-30,-10,10,30])
    ax.set_xticklabels([r'70$^{\circ}$W',r'50$^{\circ}$W',r'30$^{\circ}$W',r'10$^{\circ}$W',r'10$^{\circ}$E',r'30$^{\circ}$E'])

ax = figg.get_axes()[-2]
pos = ax.get_position()
cax = figg.add_axes([pos.x0-pos.width/4,pos.y0-0.05,pos.width*1.5,0.02])
cbar = plt.colorbar(cf,cax=cax,orientation='horizontal')
cbar.ax.set_xticklabels(np.append(denslvl[:-1],'%'))

figg.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/alltrackdens-no-sppt.png',dpi=300,bbox_inches="tight")
fig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/all-AT-SLP-no-sppt.png',dpi=300,bbox_inches="tight")
fi.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/all-MED-SLP-no-sppt.png',dpi=300,bbox_inches="tight")
plt.close('all')







