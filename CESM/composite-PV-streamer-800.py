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
from string import ascii_lowercase
import matplotlib

SIMS,ATIDS,MEDIDS = wrfsims.cesm_ids()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')
LAT = wrf.getvar(ref,'lat')

colors = ['dodgerblue','darkgreen','saddlebrown']
amps = np.array(['0.7','1.4','2.1'])

pvcmap,pvlevels,pvnorm,ticklabels=PV_cmap2()
labels=['(%s)'%x for x in ascii_lowercase]

names = ['-0-km','west','east','south','north']
km=['-0-km','200','400','800']
period=['ERA5','2010','2040','2070','2100']
labels2= ['ERA5 PD','CESM PD','CESM SC','CESM MC','CESM EC']

fullfig=plt.figure(figsize=(10,15))
fullgs=gridspec.GridSpec(nrows=5, ncols=3)
fullfig2=plt.figure(figsize=(10,15))
zl=np.arange(8000,11000,300)
for wq,perio in enumerate(period):
    for t in ['05_00']:#['03_00','03_12','04_00','04_12','05_00','05_12','06_00','06_12','07_00','07_12','08_00','08_12','09_00']:
        gfig=plt.figure(figsize=(10,3))  
        ggs = gridspec.GridSpec(nrows=1, ncols=3)
        qq = 0 
        for wl,ampl,amp in zip(range(3),['weak','moderate','strong'],amps):
            avpv = np.zeros_like(ref.variables['T'][0,0])
            avu = np.zeros_like(ref.variables['T'][0,0])
            avv = np.zeros_like(ref.variables['T'][0,0])
            avslp=np.zeros_like(ref.variables['T'][0,0])
            avz=np.zeros_like(ref.variables['T'][0,0])

         #   fig=plt.figure(figsize=(8,6))
         #   gs = gridspec.GridSpec(nrows=1, ncols=1)
         #   ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
         #   ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    
            axx = gfig.add_subplot(ggs[0,qq],projection=ccrs.PlateCarree())
            fax= fullfig.add_subplot(fullgs[wq,qq],projection=ccrs.PlateCarree())
            axx.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
            fax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
            counter = 0
            f2ax =fullfig2.add_subplot(fullgs[wq,qq],projection=ccrs.PlateCarree())
            f2ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')

            for simid,sim in enumerate(SIMS):
                medid = np.array(MEDIDS[simid])
                #print(sim)
                if not perio in sim or not amp in sim or 'wrong' in sim:
                    continue
                print(sim)
#                if np.any(medid==None):
#                    continue

                data = ds(dwrf + sim + '/wrfout_d01_2000-12-%s:00:00'%t)
                pv = wrf.getvar(data,'pvo')
#                u=wrf.getvar(data,'U')
#                v=wrf.getvar(data,'V')
                p = wrf.getvar(data,'pressure')
                slp=wrf.getvar(data,'slp')
                pv300 = wrf.interplevel(pv,p,300,meta=False)
                z=wrf.getvar(data,'z')
                z300 = wrf.interplevel(z,p,300,meta=False)

#                u300 = wrf.interplevel((u[:,:,1:]+u[:,:,:-1])/2,p,300,meta=False)
#                v300 = wrf.interplevel((v[:,1:] + v[:,:-1])/2,p,300,meta=False)
                avz+=z300
                avpv+=pv300
#                avu+=u300
#                avv+=v300
                avslp+=slp
                counter +=1

          #  hc = ax.contourf(LON[0],LAT[:,0],avpv/counter,levels=pvlevels,cmap=pvcmap,norm=pvnorm)
            
          #  ax.set_xlim(-20,40)
          #  ax.set_ylim(20,65)
    
            hb = axx.contourf(LON[0],LAT[:,0],avpv/counter,levels=pvlevels,cmap=pvcmap,norm=pvnorm)
            hz = f2ax.contourf(LON[0],LAT[:,0],avz/counter,levels=zl,cmap=matplotlib.cm.nipy_spectral)
            fax.contourf(LON[0],LAT[:,0],avpv/counter,levels=pvlevels,cmap=pvcmap,norm=pvnorm)
            tex=axx.text(-18,62,'%s %s %s'%(labels[wq*3+qq],labels2[wq],ampl),zorder=15,fontsize=8)
            tex.set_bbox(dict(facecolor='white',edgecolor='white',pad=0.2))
          #  kk=4
            cn=axx.contour(LON[0],LAT[:,0],avslp/counter,levels=np.arange(970,1030,5),colors='purple',linewidths=0.5)
            cnn=fax.contour(LON[0],LAT[:,0],avslp/counter,levels=np.arange(970,1030,5),colors='purple',linewidths=0.5)
            axx.set_xlim(-20,40)
            axx.set_ylim(20,65)
            plt.clabel(cn,inline=True,fmt='%d',fontsize=6)
            plt.clabel(cnn,inline=True,fmt='%d',fontsize=6)
            axx.set_aspect('auto')
            fax.set_aspect('auto')
          #  cbax = fig.add_axes([0, 0, 0.1, 0.1])
          #  cbar=plt.colorbar(hc, ticks=pvlevels,cax=cbax)
          #  func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.01)
          #  fig.canvas.mpl_connect('draw_event', func)
          #  cbar.ax.set_yticklabels(labels=np.append(ticklabels[:5],np.append(np.array(ticklabels[5:-1]).astype(int),'PVU')))
    
        #fig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/%s-med-streamer-composites-%s.png'%(perio,t),dpi=300,bbox_inches='tight')
        # plt.close(fig)
            tex=fax.text(-17.7,60,'%s %s %s'%(labels[wq*3+qq],labels2[wq],ampl),zorder=15,fontsize=8)

            tex.set_bbox(dict(facecolor='white',edgecolor='white',pad=0.2))
            qq+=1
            fax.set_extent([-19.5,40,20,60]) 
            if wl==0:
                fax.set_yticks([20.5,40,60])
                fax.set_yticklabels(['20$^{\circ}$N','40$^{\circ}$N','60$^{\circ}$N'])
            if perio=='2100':
                if wl==0:
                    fax.set_xticks([-19.5,0,20,40])
                    fax.set_xticklabels(['20$^{\circ}$W','0$^{\circ}$E','20$^{\circ}$E','40$^{\circ}$E'])
                else:
                    fax.set_xticks([0,20,40])
                    fax.set_xticklabels(['0$^{\circ}$E','20$^{\circ}$E','40$^{\circ}$E'])
                
    #    qk=plt.quiverkey(Q,0.8,1.025,30,r'30 m s$^{-1}$',labelpos='E')
    #    qk.set_bbox(dict(facecolor='white',edgecolor='white'))
        gfig.subplots_adjust(hspace=0,wspace=0)

        pos=axx.get_position()

        cbax = gfig.add_axes([pos.x0+pos.width, pos.y0, 0.02, pos.height])
        cbar=plt.colorbar(hb, ticks=pvlevels,cax=cbax)
        cbar.ax.set_yticklabels(labels=np.append(ticklabels[:5],np.append(np.array(ticklabels[5:-1]).astype(int),'PVU')))
        
        ax=gfig.get_axes()[0]
        ax.set_yticks([20.5,40,60])
        ax.set_yticklabels(['20$^{\circ}$N','40$^{\circ}$N','60$^{\circ}$N'])
        ax.set_xticks([-19.5,0,20,40])
        ax.set_xticklabels(['20$^{\circ}$W','0$^{\circ}$E','20$^{\circ}$E','40$^{\circ}$E']) 
        for ax in gfig.get_axes()[1:-1]:
            ax.set_xticks([0,20,40])
            ax.set_xticklabels(['0$^{\circ}$E','20$^{\circ}$E','40$^{\circ}$E'])
    
        gfig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/%s-med-streamer-composites-%s-300hPa-with-800-km.png'%(perio,t),dpi=300,bbox_inches='tight')
        plt.close(gfig)

fullfig.subplots_adjust(hspace=0,wspace=0)
axes = fullfig.get_axes()[2::3]
for q,fax in enumerate(axes):
        pos=fax.get_position()
        cbax = fullfig.add_axes([pos.x0+pos.width, pos.y0, 0.02, pos.height])
        cbar=plt.colorbar(hb, ticks=pvlevels,cax=cbax)
        if q<4:
            cbar.ax.set_yticklabels(labels=np.append(np.append('',ticklabels[1:5]),np.append(np.array(ticklabels[5:-1]).astype(int),'PVU')))
        else:
            cbar.ax.set_yticklabels(labels=np.append(ticklabels[:5],np.append(np.array(ticklabels[5:-1]).astype(int),'PVU')))

fullfig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/DJF-all-streamer-composites-%s-300hPa-with-800-km.png'%t,dpi=300,bbox_inches='tight')

plt.close('all')

for THl in [315,320,325]:
    fullfig=plt.figure(figsize=(10,15))
    fullgs=gridspec.GridSpec(nrows=5, ncols=3)
    fullfig2=plt.figure(figsize=(10,15))
    for wq,perio in enumerate(period):
        for t in ['05_00']:#['03_00','03_12','04_00','04_12','05_00','05_12','06_00','06_12','07_00','07_12','08_00','08_12','09_00']:
            gfig=plt.figure(figsize=(10,3))
            ggs = gridspec.GridSpec(nrows=1, ncols=3)
            qq = 0
            for wl,ampl,amp in zip(range(3),['weak','moderate','strong'],amps):
                avpv = np.zeros_like(ref.variables['T'][0,0])
                avu = np.zeros_like(ref.variables['T'][0,0])
                avv = np.zeros_like(ref.variables['T'][0,0])
                avslp=np.zeros_like(ref.variables['T'][0,0])
             #   fig=plt.figure(figsize=(8,6))
             #   gs = gridspec.GridSpec(nrows=1, ncols=1)
             #   ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
             #   ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    
                axx = gfig.add_subplot(ggs[0,qq],projection=ccrs.PlateCarree())
                fax= fullfig.add_subplot(fullgs[wq,qq],projection=ccrs.PlateCarree())
                axx.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
                fax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
                counter = 0
                f2ax =fullfig2.add_subplot(fullgs[wq,qq],projection=ccrs.PlateCarree())
                f2ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
                for simid,sim in enumerate(SIMS):
                    medid = np.array(MEDIDS[simid])
                    #print(sim)
                    if not perio in sim or not amp in sim or 'wrong' in sim:
                        continue
                    print(sim)
    #                if np.any(medid==None):
    #                    continue
    
                    data = ds(dwrf + sim + '/wrfout_d01_2000-12-%s:00:00'%t)
                    pv = wrf.getvar(data,'pvo')
#                    u=wrf.getvar(data,'U')
#                    v=wrf.getvar(data,'V')
#                    p = wrf.getvar(data,'pressure')
                    slp=wrf.getvar(data,'slp')
                    th=wrf.getvar(data,'th')
                    pv300 = wrf.interplevel(pv,th,THl,meta=False)
                    z=wrf.getvar(data,'z')
                    z300 = wrf.interplevel(z,th,THl,meta=False)

                    avz+=z300
                    avpv+=pv300

                    avslp+=slp
                    counter +=1
    
                hb = axx.contourf(LON[0],LAT[:,0],avpv/counter,levels=pvlevels,cmap=pvcmap,norm=pvnorm)
                hz = f2ax.contourf(LON[0],LAT[:,0],avz/counter,levels=zl,cmap=matplotlib.cm.nipy_spectral)
                fax.contourf(LON[0],LAT[:,0],avpv/counter,levels=pvlevels,cmap=pvcmap,norm=pvnorm)
                tex=axx.text(-18,62,'%s %s %s'%(labels[wq*3+qq],labels2[wq],ampl),zorder=15,fontsize=8)
                tex.set_bbox(dict(facecolor='white',edgecolor='white',pad=0.2))
              #  kk=4
                cn=axx.contour(LON[0],LAT[:,0],avslp/counter,levels=np.arange(970,1030,5),colors='purple',linewidths=0.5)
                cnn=fax.contour(LON[0],LAT[:,0],avslp/counter,levels=np.arange(970,1030,5),colors='purple',linewidths=0.5)
                axx.set_xlim(-20,40)
                axx.set_ylim(20,65)
                plt.clabel(cn,inline=True,fmt='%d',fontsize=6)
                plt.clabel(cnn,inline=True,fmt='%d',fontsize=6)
                axx.set_aspect('auto')
                fax.set_aspect('auto')
              #  cbax = fig.add_axes([0, 0, 0.1, 0.1])
              #  cbar=plt.colorbar(hc, ticks=pvlevels,cax=cbax)
              #  func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.01)
              #  fig.canvas.mpl_connect('draw_event', func)
              #  cbar.ax.set_yticklabels(labels=np.append(ticklabels[:5],np.append(np.array(ticklabels[5:-1]).astype(int),'PVU')))
    
            #fig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/%s-med-streamer-composites-%s.png'%(perio,t),dpi=300,bbox_inches='tight')
            # plt.close(fig)
                tex=fax.text(-18,62,'%s %s %s'%(labels[wq*3+qq],labels2[wq],ampl),zorder=15,fontsize=8)
    
                tex.set_bbox(dict(facecolor='white',edgecolor='white',pad=0.2))
                qq+=1
                fax.set_extent([-19.5,40,20,60])
                if wl==0:
                    fax.set_yticks([20.5,40,60])
                    fax.set_yticklabels(['20$^{\circ}$N','40$^{\circ}$N','60$^{\circ}$N'])
                if perio=='2100':
                    if wl==0:
                        fax.set_xticks([-19.5,0,20,40])
                        fax.set_xticklabels(['20$^{\circ}$W','0$^{\circ}$E','20$^{\circ}$E','40$^{\circ}$E'])
                    else:
                        fax.set_xticks([0,20,40])
                        fax.set_xticklabels(['0$^{\circ}$E','20$^{\circ}$E','40$^{\circ}$E'])
    
        #    qk=plt.quiverkey(Q,0.8,1.025,30,r'30 m s$^{-1}$',labelpos='E')
        #    qk.set_bbox(dict(facecolor='white',edgecolor='white'))
            gfig.subplots_adjust(hspace=0,wspace=0)
    
            pos=axx.get_position()
    
            cbax = gfig.add_axes([pos.x0+pos.width, pos.y0, 0.02, pos.height])
            cbar=plt.colorbar(hb, ticks=pvlevels,cax=cbax)
            cbar.ax.set_yticklabels(labels=np.append(ticklabels[:5],np.append(np.array(ticklabels[5:-1]).astype(int),'PVU')))
    
            ax=gfig.get_axes()[0]
            ax.set_yticks([20.5,40,60])
            ax.set_yticklabels(['20$^{\circ}$N','40$^{\circ}$N','60$^{\circ}$N'])
            ax.set_xticks([-19.5,0,20,40])
            ax.set_xticklabels(['20$^{\circ}$W','0$^{\circ}$E','20$^{\circ}$E','40$^{\circ}$E'])
            for ax in gfig.get_axes()[1:-1]:
                ax.set_xticks([0,20,40])
                ax.set_xticklabels(['0$^{\circ}$E','20$^{\circ}$E','40$^{\circ}$E'])
    
            gfig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/%s-med-streamer-composites-%s-%dK-with-800.png'%(perio,t,THl),dpi=300,bbox_inches='tight')
            plt.close(gfig)
    
    fullfig.subplots_adjust(hspace=0,wspace=0)
    axes = fullfig.get_axes()[2::3]
    for q,fax in enumerate(axes):
        pos=fax.get_position()
        cbax = fullfig.add_axes([pos.x0+pos.width, pos.y0, 0.02, pos.height])
        cbar=plt.colorbar(hb, ticks=pvlevels,cax=cbax)
        if q<4:
            cbar.ax.set_yticklabels(labels=np.append(np.append('',ticklabels[1:5]),np.append(np.array(ticklabels[5:-1]).astype(int),'PVU')))
        else:
            cbar.ax.set_yticklabels(labels=np.append(ticklabels[:5],np.append(np.array(ticklabels[5:-1]).astype(int),'PVU')))

    fullfig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/DJF-all-streamer-composites-%s-%dK-with-800.png'%(t,THl),dpi=300,bbox_inches='tight')
    plt.close('all')
    fullfig2.subplots_adjust(hspace=0,wspace=0)
    axes = fullfig2.get_axes()[2::3]
    for fax in axes:
            pos=fax.get_position()
            cbax = fullfig2.add_axes([pos.x0+pos.width, pos.y0, 0.02, pos.height])
            cbar=plt.colorbar(hz, ticks=zl,cax=cbax)
    
    #fullfig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/DJF-%dK-height-%s.png'%(THl,t),dpi=300,bbox_inches='tight')
    plt.close('all')
