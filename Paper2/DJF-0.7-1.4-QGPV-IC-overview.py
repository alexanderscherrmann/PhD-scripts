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
pvpos = [[93,55],[95,56],[140,73],[125,72]]
amps = [0.7,1.4,2.1,2.8]

pvcmap,pvlevels,pvnorm,pvticklabels=PV_cmap2()
ucmap,ulevels = cmocean.cm.tempo,np.arange(10,65,5)
unorm = BoundaryNorm(ulevels,ucmap.N)
PSlevel = np.arange(975,1031,5)
dis = 1500

dlat = helper.convert_radial_distance_to_lon_lat_dis_new(dis,0)

Fig = plt.figure(figsize=(12,20))

wrfd = '/atmosdyn2/ascherrmann/013-WRF-sim/'
pappath = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'
for pvpo,sea in zip(pvpos,seasons[:1]):
    ### figure

    ref = ds(wrfd + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
    
    ### unperturbed data
    PVref=wrf.getvar(ref,'pvo')
    Pref = wrf.getvar(ref,'pressure')
    Uref = ref.variables['U'][0,:]
    MSLPref = ref.variables['MSLP'][0,:]
    PV300ref = wrf.interplevel(PVref,Pref,300,meta=False)
    U300ref = wrf.interplevel((Uref[:,:,:-1] + Uref[:,:,1:])/2,Pref,300,meta=False)

    lon = wrf.getvar(ref,'lon')[0]
    lat = wrf.getvar(ref,'lat')[:,0]


    ### vertical ref
    lon_start = lon[pvpo[0]]
    lon_end=lon_start
    lat_start = lat[pvpo[1]] - dlat
    lat_end = lat_start+2*dlat

    ymin = 100.
    ymax = 1000.
    deltas = 5.
    mvcross    = Basemap()
    line,      = mvcross.drawgreatcircle(lon_start, lat_start, lon_end, lat_end, del_s=deltas)
    path       = line.get_path()
    lonp, latp = mvcross(path.vertices[:,0], path.vertices[:,1], inverse=True)
    dimpath    = len(lonp)

    Var=PVref
    vcross_pvref = np.zeros(shape=(Var.shape[0],dimpath))
    urefcross = np.zeros(shape=(Var.shape[0],dimpath))
    vcross_pref= np.zeros(shape=(Var.shape[0],dimpath))


    bottomleft = np.array([lat[0], lon[0]])
    topright   = np.array([lat[-1], lon[-1]])
    for k in range(Var.shape[0]):
        f_vcross     = Intergrid(Var[k,:,:], lo=bottomleft, hi=topright, verbose=0)
        f_p3d_vcross   = Intergrid(Pref[k,:,:], lo=bottomleft, hi=topright, verbose=0)
        f_ucross  = Intergrid(Uref[k,:,:], lo=bottomleft, hi=topright, verbose=0)
        for i in range(dimpath):
            vcross_pvref[k,i]     = f_vcross.at([latp[i],lonp[i]])
            vcross_pref[k,i]   = f_p3d_vcross.at([latp[i],lonp[i]])
            urefcross[k,i]     = f_ucross.at([latp[i],lonp[i]])

    fig=plt.figure(figsize=(16,5))
    Fig = plt.figure(figsize=(12,20))
    gs = gridspec.GridSpec(nrows=1, ncols=5)

    per = ds(wrfd + 'DJF-clim-max-U-at-300-hPa-1.4-QGPV/wrfout_d01_2000-12-01_00:00:00')

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
    text = ax.text(0.03,0.84,'(a)',transform=ax.transAxes,zorder=100)
    text.set_bbox(dict(facecolor='white',edgecolor='white'))
    ax.set_yticks([20,40,60,80])
    ax.set_yticklabels([r'20$^{\circ}$N',r'40$^{\circ}$N',r'60$^{\circ}$N',r'80$^{\circ}$N'])
    ax.set_xticks([-120,-90,-60,-30,0,30,60])
    ax.set_xticklabels([r'120$^{\circ}$W',r'90$^{\circ}$W',r'60$^{\circ}$W',r'30$^{\circ}$W',r'0$^{\circ}$E',r'30$^{\circ}$E',r'60$^{\circ}$E'])

   ### pert PV cross
    ax4=fig.add_subplot(gs[0,2:4],projection=ccrs.PlateCarree())
    ax4.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    ax4.contour(lon,lat,PV300per,levels=[2],colors='k',linewidths=2)
    hc=ax4.contourf(lon,lat,U300per,cmap=ucmap,norm=unorm,extend='max',levels=ulevels)
    ax4.set_yticks([20,40,60,79.5])
    ax4.set_yticklabels([r'20$^{\circ}$N',r'40$^{\circ}$N',r'60$^{\circ}$N',r'80$^{\circ}$N'])
    ax4.set_xticks([-119.5,-90,-60,-30,0,30,60])
    ax4.set_xticklabels([r'120$^{\circ}$W',r'90$^{\circ}$W',r'60$^{\circ}$W',r'30$^{\circ}$W',r'0$^{\circ}$E',r'30$^{\circ}$E',r'60$^{\circ}$E'])
    ax4.plot([lon_start, lon_end], [lat_start, lat_end],color='grey',linewidth=2)
#    cbax = fig.add_axes([0, 0, 0.1, 0.1])
#    cbar=plt.colorbar(hc, ticks=ulevels,cax=cbax)
#    func=resize_colorbar_vert(cbax, ax4, pad=0.0, size=0.005)
#    fig.canvas.mpl_connect('draw_event', func)
#    cbar.ax.set_yticklabels(labels=np.append(ulevels[:-1],'m s$^{-1}$'))
    ax4.set_aspect('auto')
    text = ax4.text(0.03,0.84,'(b)',transform=ax4.transAxes,zorder=100)
    text.set_bbox(dict(facecolor='white',edgecolor='white'))
    
    ### vertical U ref

    ax = fig.add_subplot(gs[0,4])
    xcoord = np.zeros(shape=(Var.shape[0],dimpath))
    for x in range(Var.shape[0]):
        xcoord[x,:] = np.array([ i*deltas-dis for i in range(dimpath) ])

    h = ax.contourf(xcoord, vcross_pper, upercross, levels = ulevels, cmap =ucmap, norm=unorm, extend = 'max')
    h1=ax.contour(xcoord,vcross_pper,upercross-urefcross,levels=np.array([2,5,10,15,20]),colors='black',linewidths=0.5)
    plt.clabel(h1,inline=True,fmt='%d',fontsize=6)
    h2=ax.contour(xcoord,vcross_pper,upercross-urefcross,levels=np.array([-20,-15,-10,-5,-2]),colors='black',linestyles='--',linewidths=0.5)
    plt.clabel(h2,inline=True,fmt='%d',fontsize=6)
    ax.contour(xcoord,vcross_pper,vcross_pvper,levels=[2],colors='purple',linewidths=1,zorder=10)
    ax.set_ylabel('Pressure [hPa]',fontsize=9)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlim(-1 * dis,dis)
    ax.set_xticks(ticks=np.arange(-1500,1100,500))
    ax.set_xticklabels(np.append('km',np.arange(-1000,1100,500)),fontsize=9)
    ax.set_yticklabels(np.arange(0,1200,200),fontsize=9)
    ax.invert_yaxis()

    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(h, ticks=ulevels,cax=cbax)
    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.005)
    fig.canvas.mpl_connect('draw_event', func)
    cbar.ax.set_yticklabels(labels=np.append(ulevels[:-1],'m s$^{-1}$'))

    ax.set_aspect('auto')
    text = ax.text(0.04,0.84,'(c)',transform=ax.transAxes,zorder=100)
    text.set_bbox(dict(facecolor='white',edgecolor='white'))

    plt.subplots_adjust(top=0.45,wspace=0.47)

    pos2 = ax4.get_position()
    pos2n = [pos2.x0+0.003,pos2.y0,pos2.width,pos2.height]
    ax4.set_position(pos2n)
    fig.savefig(pappath + 'DJF-1.4-IC-overview_contours.png',dpi=300,bbox_inches='tight')
    plt.close('all')
