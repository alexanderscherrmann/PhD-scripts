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

SIMS,ATIDS,MEDIDS = wrfsims.cesm_ids()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')
LAT = wrf.getvar(ref,'lat')

colors = ['dodgerblue','darkgreen','saddlebrown']
seas = np.array(['DJF','MAM','SON'])
amps = np.array([0.7,1.4,2.1])

pvcmap,pvlevels,pvnorm,ticklabels=PV_cmap2()
labels=['(a) C2040 weak','(b) C2040 moderate','(c) C2040 strong','(d) C2070 weak','(e) C2070 moderate','(f) C2070 strong','(g) C2100 weak','(h) C2100 moderate','(i) C2100 strong']

gfig=plt.figure(figsize=(10,5/2*3))
ggs = gridspec.GridSpec(nrows=3, ncols=3)

for qq,perio in enumerate(['2040','2070','2100']):
  for t in ['05_00']:
   for qw, amp in enumerate(amps):
    avpv = np.zeros_like(ref.variables['T'][0,0])
    avu = np.zeros_like(ref.variables['T'][0,0])
    avv = np.zeros_like(ref.variables['T'][0,0])
    avslp=np.zeros_like(ref.variables['T'][0,0])

    axx = gfig.add_subplot(ggs[qq,qw],projection=ccrs.PlateCarree())
    axx.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    counter = 0
    for simid,sim in enumerate(SIMS):
        if not perio in sim or not str(amp) in sim: 
            continue
        print(sim)
        medid = np.array(MEDIDS[simid])
        atid = np.array(ATIDS[simid])
            

        data = ds(dwrf + sim + '/wrfout_d01_2000-12-%s:00:00'%t)
        pv = wrf.getvar(data,'pvo')
        u=wrf.getvar(data,'U')
        v=wrf.getvar(data,'V')
        p = wrf.getvar(data,'pressure')
        slp=wrf.getvar(data,'slp')
        pv300 = wrf.interplevel(pv,p,300,meta=False)
        #u300 = wrf.interplevel((u[:,:,1:]+u[:,:,:-1])/2,p,300,meta=False)
        #v300 = wrf.interplevel((v[:,1:] + v[:,:-1])/2,p,300,meta=False)

        avpv+=pv300
        #avu+=u300
        #avv+=v300
        avslp+=slp
        counter +=1
    
    hb = axx.contourf(LON[0],LAT[:,0],avpv/counter,levels=pvlevels,cmap=pvcmap,norm=pvnorm)
#    kk=4
#    Q=axx.quiver(LON[0,::kk],LAT[::kk,0],avu[::kk,::kk]/counter,avv[::kk,::kk]/counter,scale=20,units='x')
    tex =axx.text(0.03,0.9,labels[3*qq+qw],transform=axx.transAxes,zorder=100)
    tex.set_bbox(dict(edgecolor='white',facecolor='white'))
    cn=axx.contour(LON[0],LAT[:,0],avslp/counter,levels=np.arange(970,1030,5),colors='purple',linewidths=0.5)
    axx.set_xlim(-20,40)
    axx.set_ylim(20,65)
    plt.clabel(cn,inline=True,fmt='%d',fontsize=6)
    axx.set_aspect('auto')

plt.subplots_adjust(hspace=0,wspace=0)

for ax in gfig.get_axes()[::3]:
    ax.set_yticks([20.5,40,60])
    ax.set_yticklabels(['20$^{\circ}$N','40$^{\circ}$N','60$^{\circ}$N'])


ax.set_xticks([-19.5,0,20,40])
ax.set_xticklabels(['20$^{\circ}$W','0$^{\circ}$E','20$^{\circ}$E','40$^{\circ}$E'])
for ax in gfig.get_axes()[-2:]:
    ax.set_xticks([0,20,40])
    ax.set_xticklabels(['0$^{\circ}$E','20$^{\circ}$E','40$^{\circ}$E'])

pos=axx.get_position()
cbax = gfig.add_axes([pos.x0+pos.width, pos.y0, 0.02, pos.height*3])
cbar=plt.colorbar(hb, ticks=pvlevels,cax=cbax)
cbar.ax.set_yticklabels(labels=np.append(ticklabels[:5],np.append(np.array(ticklabels[5:-1]).astype(int),'PVU')))

gfig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/streamers/CESM-SC-MC-EC-streamer-800km.png',dpi=300,bbox_inches='tight')
plt.close(gfig)
