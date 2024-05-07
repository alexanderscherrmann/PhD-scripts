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

period = ['ERA5','2010','2040','2070','2100']

seasons = ['DJF','MAM','SON']
x0,y0,x1,y1=70,30,181,121

pvcmap,pvlevels,pvnorm,pvticklabels=PV_cmap2()
ucmap,ulevels = cmocean.cm.tempo,np.arange(10,65,5)
unorm = BoundaryNorm(ulevels,ucmap.N)
PSlevel = np.arange(975,1031,5)
dis = 1500

dlat = helper.convert_radial_distance_to_lon_lat_dis_new(dis,0)

wrfd = '/atmosdyn2/ascherrmann/013-WRF-sim/'
pappath = '/atmosdyn2/ascherrmann/015-CESM-WRF/'
Fig = plt.figure(figsize=(12,20))
Gs = gridspec.GridSpec(nrows=4, ncols=5)
#Labels=[['(a)','(b)','(c)'],['(d)','(e)','(f)'],['(g)','(h)','(i)'],['(j)','(k)','(l)']]
#Labels=[['(a) ERA5 PD','(b) ERA5 PD','(c) ERA5 PD'],['(d) CESM PD','(e) CESM PD','(f) CESM PD'],['(g) CESM SC','(h) CESM SC','(i) CESM SC'],['(j) CESM MC','(k) CESM MC','(l) CESM MC'],['(m) CESM EC','(n) CESM EC','(o) CESM EC']]

xtext,ytext=0.03,1.03
#xtext,ytext=0.03,0.9
#for pvpo,sea in zip(pvpos,seasons):
for q,sea in enumerate(seasons):
 #for q,perio in enumerate(period):
    ### figure

    ref = ds(wrfd + '%s-clim/wrfout_d01_2000-12-01_00:00:00'%sea,'r')
    Uref = ref.variables['U'][0,:]
    Pref = wrf.getvar(ref,'pressure')
    
    per = ds(wrfd + '%s-0-km-max-2.1-QGPV-jet-check/wrfout_d01_2000-12-01_00:00:00'%sea,'r')

    lon = wrf.getvar(per,'lon')[0]
    lat = wrf.getvar(per,'lat')[:,0]
    
     ### unperturbed data
    PVper=wrf.getvar(per,'pvo')
    Pper = wrf.getvar(per,'pressure')
    Uper = per.variables['U'][0,:]
    Vper = per.variables['V'][0,:]
    THper = wrf.getvar(per,'th')

    MSLPper = per.variables['MSLP'][0,:]
    PV300per = wrf.interplevel(PVper,Pper,300,meta=False)
#    U300per = wrf.interplevel((Uper[:,:,:-1] + Uper[:,:,1:])/2,Pper,300,meta=False)
    U300per = wrf.interplevel((Uref[:,:,:-1] + Uref[:,:,1:])/2,Pref,300,meta=False)
    V300 = wrf.interplevel((Vper[:,:-1] + Vper[:,1:])/2,Pper,300,meta=False)

    ymax,xmax = np.where(U300per[y0:y1,x0:x1]==np.max(U300per[y0:y1,x0:x1]))

    #u3 = np.sqrt(V300**2 + U300per**2)

    xmax=xmax+x0
    ymax=ymax+y0
    #print(perio,xmax,ymax,x0+np.where(u3[y0:y1,x0:x1]==np.max(u3[y0:y1,x0:x1]))[1],y0+np.where(u3[y0:y1,x0:x1]==np.max(u3[y0:y1,x0:x1]))[0])

    lon_start = lon[xmax]
    lon_end=lon_start
    lat_start = lat[ymax] - dlat
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

    Var=PVper
    vcross_pvper = np.zeros(shape=(Var.shape[0],dimpath))
    thpercross = np.zeros(shape=(Var.shape[0],dimpath))
    vcross_pper= np.zeros(shape=(Var.shape[0],dimpath))
    for k in range(Var.shape[0]):
        f_vcross     = Intergrid(Var[k,:,:], lo=bottomleft, hi=topright, verbose=0)
        f_p3d_vcross   = Intergrid(Pper[k,:,:], lo=bottomleft, hi=topright, verbose=0)
        f_thcross  = Intergrid(THper[k,:,:], lo=bottomleft, hi=topright, verbose=0)
        for i in range(dimpath):
            vcross_pvper[k,i]     = f_vcross.at([latp[i],lonp[i]])
            vcross_pper[k,i]   = f_p3d_vcross.at([latp[i],lonp[i]])
            thpercross[k,i]     = f_thcross.at([latp[i],lonp[i]])


    nax = Fig.add_subplot(Gs[q,:2],projection=ccrs.PlateCarree())
    nax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    h2=nax.contour(lon,lat,MSLPper,levels=PSlevel,colors='purple',linewidths=0.5)
    Pvc=nax.contourf(lon,lat,PV300per,cmap=pvcmap,norm=pvnorm,extend='both',levels=pvlevels)
    plt.clabel(h2,inline=True,fmt='%d',fontsize=6)
    nax.set_aspect('auto')
#    text = nax.text(xtext,ytext,Labels[q][0],transform=nax.transAxes,zorder=100)

    nax4=Fig.add_subplot(Gs[q,2:4],projection=ccrs.PlateCarree())
    nax4.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    nax4.contour(lon,lat,PV300per,levels=[2],colors='k',linewidths=2)
    hc=nax4.contourf(lon,lat,U300per,cmap=ucmap,norm=unorm,extend='max',levels=ulevels)

    nax4.plot([lon_start, lon_end], [lat_start, lat_end],color='grey',linewidth=2)
    nax4.set_aspect('auto')
#    text = nax4.text(xtext,ytext,Labels[q][1],transform=nax4.transAxes,zorder=100)
    ### vertical U ref

    nax = Fig.add_subplot(Gs[q,4])
    xcoord = np.zeros(shape=(Var.shape[0],dimpath))
    for x in range(Var.shape[0]):
        xcoord[x,:] = np.array([ i*deltas-dis for i in range(dimpath) ])

    u = nax.contourf(xcoord, vcross_pper, vcross_pvper, levels = pvlevels, cmap =pvcmap, norm=pvnorm, extend = 'both')
    cn=nax.contour(xcoord,vcross_pper,thpercross,levels=np.arange(260,350,3),colors='purple',linewidths=0.5,zorder=10)
    plt.clabel(cn,inline=True,fmt='%d',fontsize=4)
    nax.set_ylabel('Pressure [hPa]')
    nax.set_ylim(bottom=ymin, top=ymax)
    nax.set_xlim(-1 * dis,dis)
    nax.set_xticks(ticks=np.arange(-1500,1600,500))
    nax.set_xticklabels(np.array([r'$\times 10^3$ km','','-0.5','0','0.5','1','1.5']))
    nax.set_yticks(np.arange(200,1200,200))
    nax.set_yticklabels(np.arange(200,1200,200))
    nax.invert_yaxis()

    nax.set_aspect('auto')
    #text = nax.text(xtext,ytext,Labels[q][2],transform=nax.transAxes,zorder=100)

Fig.subplots_adjust(top=0.5,hspace=0.25)
axes = Fig.get_axes()

for ax in axes[2::3]:
    pos = ax.get_position()
    cbax = Fig.add_axes([pos.x0+pos.width,pos.y0,0.005,pos.height])
    cbar=plt.colorbar(Pvc, ticks=pvlevels,cax=cbax)
    cbar.ax.set_yticklabels(labels=np.append(pvticklabels[:5],np.append(np.array(pvticklabels[5:-1]).astype(int),'PVU')))

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

Fig.savefig(pappath + 'ERA5-ICs-PV-TH-check.png',dpi=300,bbox_inches='tight')
plt.close('all')
