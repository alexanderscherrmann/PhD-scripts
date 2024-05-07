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

SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]

cmap,levels,norm,ticklabels=PV_cmap2()

seas = np.array(['DJF','MAM','JJA','SON'])
amps = np.array([0.7,1.4,2.1])
dis = 1000

ymin = 100.
ymax = 1000.

deltas = 5.
mvcross    = Basemap()
pappath = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'

gfig=plt.figure(figsize=(8,6))
ggs = gridspec.GridSpec(nrows=1, ncols=2)
gax=gfig.add_subplot(ggs[0,0],projection=ccrs.PlateCarree())
gax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
gax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')

ggfig,ggax=plt.subplots()
for simid,sim in enumerate(SIMS):
    if sim!='DJF-clim-max-U-at-300-hPa-1.4-QGPV' and sim!='DJF-AT-nested-1.4':
        continue

    colo='blue'
    if sim=='DJF-AT-nested-1.4':
        colo='red'
    

    medid = np.array(MEDIDS[simid])
    sea = sim[:3]

    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]

    loc = np.where(IDs==1)[0]
    lo = np.where(slp[loc] ==np.min(slp[loc]))[0][0]
    tm = t[loc[lo]]
    lonc,latc = tlon[loc[lo]],tlat[loc[lo]]

    dm =helper.simulation_time_to_day_string(tm)
    dlon = helper.convert_radial_distance_to_lon_lat_dis_new(dis,latc)

    lon_start = lonc-dlon
    lon_end   = lonc+dlon
    lat_start = latc
    lat_end   = latc

    fiiiiiig,ax=plt.subplots()
    line,      = mvcross.drawgreatcircle(lon_start, lat_start, lon_end, lat_end, del_s=deltas)
    path       = line.get_path()
    lonp, latp = mvcross(path.vertices[:,0], path.vertices[:,1], inverse=True)
    dimpath    = len(lonp)

    data = ds(dwrf + sim + '/wrfout_d01_2000-12-%s:00:00'%dm)
    pv = wrf.getvar(data,'pvo')
    P = wrf.getvar(data,'pressure')

    Var = pv 
    vcross = np.zeros(shape=(Var.shape[0],dimpath))
    vcross_p= np.zeros(shape=(Var.shape[0],dimpath))
    bottomleft = np.array([LAT[0], LON[0]])
    topright   = np.array([LAT[-1], LON[-1]])


    for k in range(Var.shape[0]):
        f_vcross     = Intergrid(Var[k,:,:], lo=bottomleft, hi=topright, verbose=0)
        f_p3d_vcross   = Intergrid(P[k,:,:], lo=bottomleft, hi=topright, verbose=0)
        for i in range(dimpath):
            vcross[k,i]     = f_vcross.at([latp[i],lonp[i]])
            vcross_p[k,i]   = f_p3d_vcross.at([latp[i],lonp[i]])

    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(nrows=1, ncols=2)
    ax = fig.add_subplot(gs[0,0])
    ax.plot(t[loc]/24,slp[loc],color='k')

    gax.plot(tlon[loc],tlat[loc],color=colo)
    data2=ds(dwrf+sim+'/wrfout_d01_2000-12-04_12:00:00')
    dpv=wrf.getvar(data2,'pvo')
    dpres=wrf.getvar(data2,'pressure')
    dpv300=wrf.interplevel(dpv,dpres,300,meta=False)

    gax.contour(LON,LAT,dpv300,levels=[2],colors='purple')
    if sim!='DJF-AT-nested-1.4':
        deltapv=dpv300
    else:
        deltapv-=dpv300

    ggax.plot(t[loc]/24,slp[loc],color=colo)
    ax.set_xlabel('simulation time [h]')
    ax.set_xticks(ticks=np.arange(1,9,1))
    ax.set_xlim(t[loc[0]]/24,t[loc[-1]]/24)

    ax.set_ylabel('SLP [hPa]')
    ax.set_ylim(975,1010)
    tex=ax.text(0.03,0.93,'(d)',transform=ax.transAxes)
    tex.set_zorder(102)
    tex.set_bbox(dict(facecolor='white',edgecolor='white'))
    ax = fig.add_subplot(gs[0,1])

    xcoord = np.zeros(shape=(Var.shape[0],dimpath))
    for x in range(Var.shape[0]):
        xcoord[x,:] = np.array([ i*deltas-dis for i in range(dimpath) ])

    h = ax.contourf(xcoord, vcross_p, vcross, levels = levels, cmap = cmap, norm=norm, extend = 'both')

    ax.set_ylabel('Pressure [hPa]', fontsize=12)
    ax.set_ylim(bottom=ymin, top=ymax)

    ax.set_xlim(-1000,1000)
    ax.set_xticks(ticks=np.arange(-1000,1001,250))
    ax.set_xticklabels(labels=np.append(np.arange(-1000,1000,250),'km'))
    ax.invert_yaxis()

    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(h, ticks=levels,cax=cbax)
    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.01)
    cbar.ax.set_yticklabels(labels=np.append(ticklabels[:5],np.append(np.array(ticklabels[5:-1]).astype(int),'PVU')))
    fig.canvas.mpl_connect('draw_event', func)
    tex=ax.text(0.03,0.93,'(e)',transform=ax.transAxes)
    tex.set_bbox(dict(facecolor='white',edgecolor='white'))

    plt.subplots_adjust(top=0.7)
    fig.savefig(pappath + '/DJF-1.4-AT-cyclone-SLP-evo-PVcross-%s.png'%sim,dpi=300,bbox_inches='tight')
    plt.close(fig)

    loc=np.where(IDs==2)[0]
    ggax.plot(t[loc]/24,slp[loc],color=colo)
    gax.plot(tlon[loc],tlat[loc],color=colo)


levels=np.arange(-1.5,1.6,0.3)
hb=gax.contourf(LON,LAT,deltapv,cmap=matplotlib.cm.coolwarm,levels=levels)
gax.set_xlim(-80,40)
gax.set_ylim(20,70)
cbax = gfig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(hb, ticks=levels,cax=cbax)
func=resize_colorbar_vert(cbax, gax, pad=0.0, size=0.01)
cbar.ax.set_yticklabels(labels=np.append(np.append('$\Delta$PV',np.round(levels[1:-1],2)),'PVU'))
gfig.canvas.mpl_connect('draw_event', func)

gfig.savefig(pappath+'/DJF-1.4-AT-nested-comparison.png',dpi=300,bbox_inches='tight')
ggfig.savefig(pappath+'/DJF-1.4-SLP-nested-comparison.png',dpi=300,bbox_inches='tight')
plt.close('all')

