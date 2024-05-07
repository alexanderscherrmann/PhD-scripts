import numpy as np
from netCDF4 import Dataset as ds
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy
from matplotlib.colors import BoundaryNorm
import wrf
import sys
import cmocean

sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
from colormaps import PV_cmap2

from dypy.intergrid import Intergrid
from mpl_toolkits.basemap import Basemap
import helper

seasons = ['DJF','MAM','JJA','SON']
labels = ['(a)','(b)','(c)']

pvpo = [93,55]

pvcmap,pvlevels,pvnorm,pvticklabels=PV_cmap2()
ucmap,ulevels = cmocean.cm.tempo,np.arange(10,80,5)
unorm = BoundaryNorm(ulevels,ucmap.N)
PSlevel = np.arange(975,1031,5)
dis = 1500

dlat = helper.convert_radial_distance_to_lon_lat_dis_new(dis,0)

wrfd = '/atmosdyn2/ascherrmann/013-WRF-sim/'
pappath = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'
Fig = plt.figure(figsize=(12,15))
Gs = gridspec.GridSpec(nrows=3, ncols=5)
#Labels=[['(a)','(b)','(c)'],['(d)','(e)','(f)'],['(g)','(h)','(i)'],['(j)','(k)','(l)']]
Labels=[['(a) winter 4N moderate','(b) winter 4N moderate','(c) winter 4N moderate'],['(d) winter 4W moderate','(e) winter 4W moderate','(f) winter 4W moderate'],['(g) winter 4N strong','(h) winter 4N strong','(i) winter 4N strong'],['(j) winter 4N strong','(k) winter 4N strong','(l) winter 4N strong']]
sims=['DJF-400-km-north-from-max-300-hPa-1.4-QGPV','DJF-400-km-west-from-max-300-hPa-1.4-QGPV','DJF-400-km-north-from-max-300-hPa-2.1-QGPV']

xtext,ytext=0.03,1.03
adds=[[0,8],[-8,0],[0,8]]
#xtext,ytext=0.03,0.9
for ad,(q,sim) in zip(adds,enumerate(sims)):
    ### figure

    per = ds(wrfd + sim + '/wrfout_d01_2000-12-01_00:00:00','r')
    lon = wrf.getvar(per,'lon')[0]
    lat = wrf.getvar(per,'lat')[:,0]
    
    lon_start = lon[pvpo[0]+ad[0]]
    lon_end=lon_start
    lat_start = lat[pvpo[1]+ad[0]] - dlat
    lat_end = lat_start+2*dlat


    ymin = 100.
    ymax = 1000.
    deltas = 5.
    fifig = plt.figure(figsize=(8,6))
    mvcross    = Basemap()
    line,      = mvcross.drawgreatcircle(lon_start, lat_start, lon_end, lat_end, del_s=deltas)
    path       = line.get_path()
    lonp, latp = mvcross(path.vertices[:,0], path.vertices[:,1], inverse=True)
    dimpath    = len(lonp)
    bottomleft = np.array([lat[0], lon[0]])
    topright   = np.array([lat[-1], lon[-1]])

    fig=plt.figure(figsize=(16,5))
    gs = gridspec.GridSpec(nrows=1, ncols=5)
    AXES=[]
    ### unperturbed data
    PVper=wrf.getvar(per,'pvo')
    Pper = wrf.getvar(per,'pressure')
    Uper = per.variables['U'][0,:]
    MSLPper = per.variables['MSLP'][0,:]
    PV300per = wrf.interplevel(PVper,Pper,300,meta=False)
    U300per = wrf.interplevel((Uper[:,:,:-1] + Uper[:,:,1:])/2,Pper,300,meta=False)
#
#
    Var=PVper
    vcross_pvper = np.zeros(shape=(Var.shape[0],dimpath))
    upercross = np.zeros(shape=(Var.shape[0],dimpath))
    vcross_pper= np.zeros(shape=(Var.shape[0],dimpath))
    for k in range(Var.shape[0]):
        f_vcross     = Intergrid(Var[k,:,:], lo=bottomleft, hi=topright, verbose=0)
        f_p3d_vcross   = Intergrid(Pper[k,:,:], lo=bottomleft, hi=topright, verbose=0)
        f_ucross  = Intergrid(Uper[k,:,:], lo=bottomleft, hi=topright, verbose=0)
        for i in range(dimpath):
            vcross_pvper[k,i]     = f_vcross.at([latp[i],lonp[i]])
            vcross_pper[k,i]   = f_p3d_vcross.at([latp[i],lonp[i]])
            upercross[k,i]     = f_ucross.at([latp[i],lonp[i]])


   ### ref PV cross
    ax = fig.add_subplot(gs[0,:2],projection=ccrs.PlateCarree())
    AXES.append(ax)
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')

    h2=ax.contour(lon,lat,MSLPper,levels=PSlevel,colors='purple',linewidths=0.5)
    hc=ax.contourf(lon,lat,PV300per,cmap=pvcmap,norm=pvnorm,extend='both',levels=pvlevels)
    plt.clabel(h2,inline=True,fmt='%d',fontsize=6)
    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(hc, ticks=pvlevels,cax=cbax)
    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.005)
    fig.canvas.mpl_connect('draw_event', func)
    cbar.ax.set_yticklabels(labels=np.append(pvticklabels[:5],np.append(np.array(pvticklabels[5:-1]).astype(int),'PVU')))
    ax.set_aspect('auto')
    text = ax.text(xtext,ytext,labels[0],transform=ax.transAxes,zorder=100)
#    text.set_bbox(dict(facecolor='white',edgecolor='white'))

    nax = Fig.add_subplot(Gs[q,:2],projection=ccrs.PlateCarree())
    nax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    h2=nax.contour(lon,lat,MSLPper,levels=PSlevel,colors='purple',linewidths=0.5)
    Pvc=nax.contourf(lon,lat,PV300per,cmap=pvcmap,norm=pvnorm,extend='both',levels=pvlevels)
    plt.clabel(h2,inline=True,fmt='%d',fontsize=6)
    #cbax = Fig.add_axes([0, 0, 0.1, 0.1])
    #cbar=plt.colorbar(hc, ticks=pvlevels,cax=cbax)
    #func=resize_colorbar_vert(cbax, nax, pad=0.0, size=0.005)
    #fig.canvas.mpl_connect('draw_event', func)
    #cbar.ax.set_yticklabels(labels=np.append(pvticklabels[:5],np.append(np.array(pvticklabels[5:-1]).astype(int),'PVU')))
    nax.set_aspect('auto')
    text = nax.text(xtext,ytext,Labels[q][0],transform=nax.transAxes,zorder=100)
#    text.set_bbox(dict(facecolor='white',edgecolor='white'))

    ### pert PV cross
    ax4=fig.add_subplot(gs[0,2:4],projection=ccrs.PlateCarree())
    AXES.append(ax4)
    ax4.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    ax4.contour(lon,lat,PV300per,levels=[2],colors='k',linewidths=2)
    hc=ax4.contourf(lon,lat,U300per,cmap=ucmap,norm=unorm,extend='max',levels=ulevels)

    ax4.plot([lon_start, lon_end], [lat_start, lat_end],color='grey',linewidth=2)
    ax4.set_aspect('auto')
    text = ax4.text(xtext,ytext,labels[1],transform=ax4.transAxes,zorder=100)
#    text.set_bbox(dict(facecolor='white',edgecolor='white'))

    nax4=Fig.add_subplot(Gs[q,2:4],projection=ccrs.PlateCarree())
    nax4.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    nax4.contour(lon,lat,PV300per,levels=[2],colors='k',linewidths=2)
    hc=nax4.contourf(lon,lat,U300per,cmap=ucmap,norm=unorm,extend='max',levels=ulevels)

    nax4.plot([lon_start, lon_end], [lat_start, lat_end],color='grey',linewidth=2)
    nax4.set_aspect('auto')
    text = nax4.text(xtext,ytext,Labels[q][1],transform=nax4.transAxes,zorder=100)
#    text.set_bbox(dict(facecolor='white',edgecolor='white'))

    ### vertical U ref

    ax = fig.add_subplot(gs[0,4])
    AXES.append(ax)
    xcoord = np.zeros(shape=(Var.shape[0],dimpath))
    for x in range(Var.shape[0]):
        xcoord[x,:] = np.array([ i*deltas-dis for i in range(dimpath) ])

    h = ax.contourf(xcoord, vcross_pper, upercross, levels = ulevels, cmap =ucmap, norm=unorm, extend = 'max')
    ax.contour(xcoord,vcross_pper,vcross_pvper,levels=[2],colors='purple',linewidths=1,zorder=10)
    ax.set_ylabel('Pressure [hPa]',fontsize=9)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlim(-1 * dis,dis)
    ax.set_xticks(ticks=np.arange(-1500,1100,500))
    ax.set_xticklabels(np.append('km',np.arange(-1000,1100,500)),fontsize=9)
    ax.set_yticks(np.arange(200,1200,200))
    ax.set_yticklabels(np.arange(200,1200,200),fontsize=9)
    ax.invert_yaxis()

    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(h, ticks=ulevels,cax=cbax)
    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.005)
    fig.canvas.mpl_connect('draw_event', func)
    cbar.ax.set_yticklabels(labels=np.append(ulevels[:-1],'m s$^{-1}$'))

    ax.set_aspect('auto')
    text = ax.text(xtext,ytext,labels[2],transform=ax.transAxes,zorder=100)
#    text.set_bbox(dict(facecolor='white',edgecolor='white'))
    
    nax = Fig.add_subplot(Gs[q,4])
    xcoord = np.zeros(shape=(Var.shape[0],dimpath))
    for x in range(Var.shape[0]):
        xcoord[x,:] = np.array([ i*deltas-dis for i in range(dimpath) ])

    u = nax.contourf(xcoord, vcross_pper, upercross, levels = ulevels, cmap =ucmap, norm=unorm, extend = 'max')
    nax.contour(xcoord,vcross_pper,vcross_pvper,levels=[2],colors='purple',linewidths=1,zorder=10)
    nax.set_ylabel('Pressure [hPa]')
    nax.set_ylim(bottom=ymin, top=ymax)
    nax.set_xlim(-1 * dis,dis)
    nax.set_xticks(ticks=np.arange(-1500,1600,500))
    nax.set_xticklabels(np.array([r'$\times 10^3$ km','','-0.5','0','0.5','1','1.5']))
#    nax.set_xticklabels(np.append('km',np.arange(-1000,1100,500)),fontsize=9)
    nax.set_yticks(np.arange(200,1200,200))
    nax.set_yticklabels(np.arange(200,1200,200))
    nax.invert_yaxis()

#    cbax = Fig.add_axes([0, 0, 0.1, 0.1])
#    cbar=plt.colorbar(h, ticks=ulevels,cax=cbax)
#    func=resize_colorbar_vert(cbax, nax, pad=0.0, size=0.005)
#    fig.canvas.mpl_connect('draw_event', func)
#    cbar.ax.set_yticklabels(labels=np.append(ulevels[:-1],'m s$^{-1}$'))

    nax.set_aspect('auto')
    text = nax.text(xtext,ytext,Labels[q][2],transform=nax.transAxes,zorder=100)
#    text.set_bbox(dict(facecolor='white',edgecolor='white'))

#    plt.subplots_adjust(top=0.45,wspace=0.4)

    fig.subplots_adjust(top=0.45)
#    for ax in AXES[2:]:
#        pos = ax.get_position()
#        cbax = Fig.add_axes([pos.x0+pos.width,pos.y0,0.005,pos.height])
#        cbar=plt.colorbar(h, ticks=ulevels,cax=cbax)
#        cbar.ax.set_yticklabels(labels=np.append(ulevels[:-1],'m s$^{-1}$'))
#    
#    for ax in AXES[:1]:
#        pos = ax.get_position()
#        ax.set_position([pos.x0-0.03,pos.y0,pos.width-0.03,pos.height])
#        cbax = Fig.add_axes([pos.x0-0.03+pos.width-0.03,pos.y0,0.005,pos.height])
#        cbar=plt.colorbar(Pvc, ticks=pvlevels,cax=cbax)
#        cbar.ax.set_yticklabels(labels=np.append(pvticklabels[:5],np.append(np.array(pvticklabels[5:-1]).astype(int),'PVU')))
#        ax.set_yticks([20,40,60,80])
#        ax.set_yticklabels([r'20$^{\circ}$N',r'40$^{\circ}$N',r'60$^{\circ}$N',r'80$^{\circ}$N'])
#        ax.set_xticks([-120,-90,-60,-30,0,30,60])
#        ax.set_xticklabels([r'120$^{\circ}$W',r'90$^{\circ}$W',r'60$^{\circ}$W',r'30$^{\circ}$W',r'0$^{\circ}$E',r'30$^{\circ}$E',r'60$^{\circ}$E'])
#    
#    for ax in AXES[1:2]:
#        pos = ax.get_position()
#        ax.set_position([pos.x0,pos.y0,pos.width-0.03,pos.height])
#        ax.set_yticks([20,40,60,80])
#        ax.set_yticklabels([r'20$^{\circ}$N',r'40$^{\circ}$N',r'60$^{\circ}$N',r'80$^{\circ}$N'])
#        ax.set_xticks([-120,-90,-60,-30,0,30,60])
#        ax.set_xticklabels([r'120$^{\circ}$W',r'90$^{\circ}$W',r'60$^{\circ}$W',r'30$^{\circ}$W',r'0$^{\circ}$E',r'30$^{\circ}$E',r'60$^{\circ}$E'])

#    fig.savefig(pappath + sea + 'clim-IC-overview.png',dpi=300,bbox_inches='tight')
    plt.close(fig)
    plt.close(fifig)

Fig.subplots_adjust(top=0.5,hspace=0.25)
axes = Fig.get_axes()

for ax in axes[2::3]:
    pos = ax.get_position()
    cbax = Fig.add_axes([pos.x0+pos.width,pos.y0,0.005,pos.height])
    cbar=plt.colorbar(h, ticks=ulevels,cax=cbax)
    cbar.ax.set_yticklabels(labels=np.append(ulevels[:-1],'m s$^{-1}$'))

for ax in axes[::3]:
    pos = ax.get_position()
    ax.set_position([pos.x0-0.03,pos.y0,pos.width-0.03,pos.height])
    cbax = Fig.add_axes([pos.x0-0.03+pos.width-0.03,pos.y0,0.005,pos.height])
    cbar=plt.colorbar(Pvc, ticks=pvlevels,cax=cbax)
    cbar.ax.set_yticklabels(labels=np.append(pvticklabels[:5],np.append(np.array(pvticklabels[5:-1]).astype(int),'PVU')))
    ax.set_yticks([20,40,60,80])
    ax.set_yticklabels([r'20$^{\circ}$N',r'40$^{\circ}$N',r'60$^{\circ}$N',r'80$^{\circ}$N'])
    ax.set_xticks([-120,-90,-60,-30,0,30,60])
    ax.set_xticklabels([r'120$^{\circ}$W',r'90$^{\circ}$W',r'60$^{\circ}$W',r'30$^{\circ}$W',r'0$^{\circ}$E',r'30$^{\circ}$E',r'60$^{\circ}$E'])

for ax in axes[1::3]:
    pos = ax.get_position()
    ax.set_position([pos.x0,pos.y0,pos.width-0.03,pos.height])
    ax.set_yticks([20,40,60,80])
    ax.set_yticklabels([r'20$^{\circ}$N',r'40$^{\circ}$N',r'60$^{\circ}$N',r'80$^{\circ}$N'])
    ax.set_xticks([-120,-90,-60,-30,0,30,60])
    ax.set_xticklabels([r'120$^{\circ}$W',r'90$^{\circ}$W',r'60$^{\circ}$W',r'30$^{\circ}$W',r'0$^{\circ}$E',r'30$^{\circ}$E',r'60$^{\circ}$E'])

Fig.savefig(pappath + 'review-1-IC-overview.png',dpi=300,bbox_inches='tight')
plt.close('all')
