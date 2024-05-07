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
labels=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)','(m)','(n)','(o)']
strength=['weak','moderate','strong']
labels2= ['ERA5 PD','CESM PD','CESM SC','CESM MC','CESM EC']
figg=plt.figure(figsize=(10,9))
fig=plt.figure(figsize=(10,9))
fi=plt.figure(figsize=(10,9))

gs=gridspec.GridSpec(nrows=5, ncols=3)


lab = np.arange(3,217,3)
sea='DJF'
#for row,sea in enumerate(['DJF','MAM','SON']):
f = open(dwrf + 'data/CESM-density-data-slp-%s.txt'%sea,'rb')
d = pickle.load(f)
f.close()
trackdens = dict()
for row,perio in enumerate(['ERA5','2010','2040','2070','2100']):
    trackdens[perio] = dict()

    atc = d['atcount']
    atdens = d['atdens']
    atslp = d['atslp']
    attrd = d['attrackdens']
    
    medc = d['medcount']
    meddens = d['meddens']
    medslp = d['medslp']
    medtrd = d['medtrackdens']
    
    for col,am in enumerate(['0.7','1.4','2.1']):
        count = 0
        alltrackdens = np.zeros_like(atdens[perio][am])
        simcount = 0
        print(atc[perio][am].keys())
        print(attrd[perio][am].keys())
        for k in atc[perio][am].keys():
            if am in k:
                try:
                    count+=atc[perio][am][k]

#                if sea=='DJF' and am=='0.7' and np.any(medtrd[perio][am][k][35:64,240:261]!=0):
#                    medtrd[perio][am][k][35:64,230:261]=0
                    alltrackdens+=attrd[perio][am][k]
                    alltrackdens+=medtrd[perio][am][k]
                    simcount+=1
                except:
                    continue

    
        alltrackdens/=simcount
        trackdens[perio][am] = alltrackdens
        meanbox = []
        for tt in atslp[perio][am].keys():
            meanbox.append(np.sort(atslp[perio][am][tt]))

        ax=figg.add_subplot(gs[row,col],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
        alltrackdens[alltrackdens>1]=1
        cf = ax.contourf(LON,LAT,alltrackdens*100,cmap=cmap,levels=denslvl,norm=norm)

        tex = ax.text(0.015,0.9,'%s %d %s %s'%(labels[3*row+col],len(list(attrd[perio][am].keys())),labels2[row],strength[col]),transform=ax.transAxes,zorder=1000)

        tex.set_bbox(dict(edgecolor='white',facecolor='white',pad=0.2))
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

        if perio=='ERA5':
            tex = axx.text(0.4,0.1,'%s %s %s'%(labels[3*row+col],labels2[row],strength[col]),transform=axx.transAxes,zorder=1000)
        else:
            tex = axx.text(0.4,0.1,'%s %s %s'%(labels[3*row+col],labels2[row],strength[col]),transform=axx.transAxes,zorder=1000)

        tex.set_bbox(dict(edgecolor='white',facecolor='white'))
        axx.set_aspect('auto')
    
        meanbox = []
        for tt in medslp[perio][am].keys():
            meanbox.append(np.sort(medslp[perio][am][tt]))
    
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

        axxx.set_ylim(990,1025)
        axxx.set_xticks(ticks=np.arange(4,73,4))
        if row==2:
            axxx.set_xticklabels(labels=np.ravel([np.array(['' for x in np.arange(1,10,1)]),(np.arange(1,19,1)[1::2]/2).astype(int)],'F'))
            axxx.set_xlabel('simulation time [d]')
        else:
            axxx.set_xticklabels([])
        if perio=='ERA5':
            tex = axxx.text(0.03,0.1,'%s %s %s'%(labels[3*row+col],labels2[row],strength[col]),transform=axxx.transAxes,zorder=1000)
        else:
            tex = axxx.text(0.03,0.1,'%s %s %s'%(labels[3*row+col],labels2[row],strength[col]),transform=axxx.transAxes,zorder=1000)
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

figg.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/images/alltrackdens-%s.png'%sea,dpi=300,bbox_inches="tight")
fig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/images/all-AT-SLP-%s.png'%sea,dpi=300,bbox_inches="tight")
fi.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/images/all-MED-SLP-%s.png'%sea,dpi=300,bbox_inches="tight")
plt.close('all')
