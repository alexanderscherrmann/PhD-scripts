import numpy as np
import netCDF4
import argparse
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy

import wrf

import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from colormaps import PV_cmap2

from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert

def readcdf(ncfile,varnam):
    infile = netCDF4.Dataset(ncfile, mode='r')
    var = infile.variables[varnam][:]
    return(var)

sims=['DJF-clim-max-U-at-300-hPa-1.4-QGPV']

fb = 'wrfout_d01_2000-12-'
sb = 'PV-300-2000-'
fe = ':00:00'

#latpairs=[[35,60],[35,65],[35,70],[40,60],[40,65],[40,70]]
latpairs=[[35,65]]

amps = ['0.7-QGPV','1.4-QGPV','2.1-QGPV']
km = ['-clim-max-U-at-300-hPa-','from-max-300-hPa-']
dis= ['-200-km-','-400-km-']
disp = ['east-','north-','south-','west-']
seasons=['DJF']

sims=[]
for sea in seasons:
    for amp in amps:
        sims.append(sea + km[0] + amp)

    for amp in amps:
        for di in dis:
            for po in disp:
                sims.append(sea + di + po + km[1] + amp)

ampav = dict()
ampcount=dict()
print(sims)
#cbax.set_yticklabels(np.append(np.arange(-50,41,10),'mm (9 d)$^{-1}$'))
Vlevel = np.arange(-25,26,5)
cmap = matplotlib.cm.coolwarm

norm = plt.Normalize(np.min(Vlevel),np.max(Vlevel))
ticklabels=Vlevel
levels=Vlevel

for qt,sim in enumerate(sims):
    p = '/atmosdyn2/ascherrmann/013-WRF-sim/' + sim + '/'
    
    for d in range(1,11):
        for h in ['00','06','12','18']:
            
            f = p + fb + '%02d_'%d + h + fe
            
            try:
                data = netCDF4.Dataset(f)
            except:
                tav['35,65']=np.vstack((tav['35,65'],np.zeros_like(tav['35,65'][0,:])))

            V = wrf.getvar(data,'V') #in PVU already
            V=(V[:,:-1]+V[:,1:])/2
            pres = wrf.getvar(data,'pressure')
            v300 = wrf.interplevel(V,pres,300,meta=False)
    
            lon = wrf.getvar(data,'lon')[0]
            lat = wrf.getvar(data,'lat')[:,0]
            if d==1 and h=='00':
                tav=dict()
                for lala in latpairs:

                    tav['%d,%d'%(lala[0],lala[1])]=np.mean(v300[np.where((lat>=lala[0])&(lat<=lala[1]))[0][0]:np.where((lat>=lala[0])&(lat<=lala[1]))[0][-1]],axis=0)
            else:
                for lala in latpairs:
                    tav['%d,%d'%(lala[0],lala[1])]=np.vstack((tav['%d,%d'%(lala[0],lala[1])],np.mean(v300[np.where((lat>=lala[0])&(lat<=lala[1]))[0][0]:np.where((lat>=lala[0])&(lat<=lala[1]))[0][-1]],axis=0)))
    
            if d==10:
                break
    if qt==0:
        ovallav=tav['35,65']
        ovc=1
        for amp in amps:
            ampav[amp]=np.zeros_like(tav['35,65'])
            ampcount[amp]=0
    else:
        ovc+=1
        ovallav+=tav['35,65']

    for amp in amps:
        if amp in sim:
            ampcount[amp]+=1
            ampav[amp]+=tav['35,65']


    for lala in latpairs:
        fig = plt.figure(figsize=(6,7))
        gs = gridspec.GridSpec(nrows=5, ncols=1)
    
        ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
        ax.set_xlim(-100,60)
        ax.set_ylim(lala[0],lala[1])

        ax.set_yticks([40,60])
        ax.set_yticklabels([r'40$^{\circ}$N',r'60$^{\circ}$N'])
    
        ax=fig.add_subplot(gs[1:4,0])
    
    
        t=np.ones_like(tav['35,65'][:,0])*6
        t[0]=0
        t=np.cumsum(t)
    
        hc=ax.contourf(lon,t,tav['%d,%d'%(lala[0],lala[1])],cmap=cmap,levels=Vlevel,extend='both')
        ax.invert_yaxis()
    
        ax.set_xticks([-120,-90,-60,-30,0,30,60])
        ax.set_xticklabels([r'120$^{\circ}$W',r'90$^{\circ}$W',r'60$^{\circ}$W',r'30$^{\circ}$W',r'0$^{\circ}$E',r'30$^{\circ}$E',r'60$^{\circ}$E'])
        ax.set_xlim(-100,60)
    
        ax.set_yticks(ticks=np.arange(0,217,24))
        ax.set_yticklabels(labels=(np.arange(0,217,24)/24).astype(int))
    
        plt.subplots_adjust(hspace=0)
    
        ax=fig.get_axes()[-1]
        pos=ax.get_position()
    
        cbax = fig.add_axes([pos.x0+pos.width, pos.y0, 0.02, pos.height])
        cbar=plt.colorbar(hc, ticks=Vlevel,cax=cbax)
    
        cbar.ax.set_yticklabels(labels=np.append(Vlevel[:-1].astype(int),'m s$^{-1}$'))	
    
        sa = p + 'hovmoeller-v-%d-%d.png'%(lala[0],lala[1])
        fig.savefig(sa,dpi=300,bbox_inches="tight")
        plt.close(fig)

for amp in amps:
    fig = plt.figure(figsize=(6,7))
    gs = gridspec.GridSpec(nrows=5, ncols=1)
    
    ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    ax.set_xlim(-100,60)
    ax.set_ylim(lala[0],lala[1])
    
    ax.set_yticks([40,60])
    ax.set_yticklabels([r'40$^{\circ}$N',r'60$^{\circ}$N'])
    
    ax=fig.add_subplot(gs[1:4,0])
    
    
    t=np.ones_like(tav['35,65'][:,0])*6
    t[0]=0
    t=np.cumsum(t)
    
    hc=ax.contourf(lon,t,ampav[amp]/ampcount[amp],cmap=cmap,levels=Vlevel,extend='both')
    ax.invert_yaxis()
    
    ax.set_xticks([-120,-90,-60,-30,0,30,60])
    ax.set_xticklabels([r'120$^{\circ}$W',r'90$^{\circ}$W',r'60$^{\circ}$W',r'30$^{\circ}$W',r'0$^{\circ}$E',r'30$^{\circ}$E',r'60$^{\circ}$E'])
    ax.set_xlim(-100,60)
    
    ax.set_yticks(ticks=np.arange(0,217,24))
    ax.set_yticklabels(labels=(np.arange(0,217,24)/24).astype(int))
    
    plt.subplots_adjust(hspace=0)
    
    ax=fig.get_axes()[-1]
    pos=ax.get_position()
    
    cbax = fig.add_axes([pos.x0+pos.width, pos.y0, 0.02, pos.height])
    cbar=plt.colorbar(hc, ticks=Vlevel,cax=cbax)
    
    cbar.ax.set_yticklabels(labels=np.append(Vlevel[:-1].astype(int),'m s$^{-1}$'))
    
    sa = '/atmosdyn2/ascherrmann/paper/NA-MED-link/hovmoeller-v-av-%s.png'%(amp)
    fig.savefig(sa,dpi=300,bbox_inches="tight")
    plt.close(fig)


fig = plt.figure(figsize=(6,7))
gs = gridspec.GridSpec(nrows=5, ncols=1)

ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
ax.set_xlim(-100,60)
ax.set_ylim(lala[0],lala[1])

ax.set_yticks([40,60])
ax.set_yticklabels([r'40$^{\circ}$N',r'60$^{\circ}$N'])

ax=fig.add_subplot(gs[1:4,0])


t=np.ones_like(tav['35,65'][:,0])*6
t[0]=0
t=np.cumsum(t)

hc=ax.contourf(lon,t,ovallav/ovc,cmap=cmap,levels=Vlevel,extend='both')
ax.invert_yaxis()

ax.set_xticks([-120,-90,-60,-30,0,30,60])
ax.set_xticklabels([r'120$^{\circ}$W',r'90$^{\circ}$W',r'60$^{\circ}$W',r'30$^{\circ}$W',r'0$^{\circ}$E',r'30$^{\circ}$E',r'60$^{\circ}$E'])
ax.set_xlim(-100,60)

ax.set_yticks(ticks=np.arange(0,217,24))
ax.set_yticklabels(labels=(np.arange(0,217,24)/24).astype(int))

plt.subplots_adjust(hspace=0)

ax=fig.get_axes()[-1]
pos=ax.get_position()

cbax = fig.add_axes([pos.x0+pos.width, pos.y0, 0.02, pos.height])
cbar=plt.colorbar(hc, ticks=Vlevel,cax=cbax)

cbar.ax.set_yticklabels(labels=np.append(Vlevel[:-1].astype(int),'m s$^{-1}$'))

sa = '/atmosdyn2/ascherrmann/paper/NA-MED-link/hovmoeller-v-av.png'
fig.savefig(sa,dpi=300,bbox_inches="tight")
plt.close(fig)

