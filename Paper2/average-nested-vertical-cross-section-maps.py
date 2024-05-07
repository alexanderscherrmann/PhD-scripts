import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from netCDF4 import Dataset as ds
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
import wrf

from dypy.intergrid import Intergrid
from mpl_toolkits.basemap import Basemap
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
import helper
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import wrfsims

SIMS,ATIDS,MEDIDS,MEDIDS2 = wrfsims.nested_ids()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
spaper = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'

tracks = '/atmosdyn2/ascherrmann/scripts/WRF/nested-cyclone-tracking/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]

cmap,levels,norm,ticklabels=PV_cmap2()
dis=500
deltas=5

mvcross    = Basemap()
initl, = mvcross.drawgreatcircle(-1 * helper.convert_radial_distance_to_lon_lat_dis_new(dis,0),0,helper.convert_radial_distance_to_lon_lat_dis_new(dis,0),0,del_s=deltas)
initpa = initl.get_path()
initlop,initlap = mvcross(initpa.vertices[:,0],initpa.vertices[:,1], inverse=True)
initdim = len(initlop)


seas = np.array(['DJF','MAM','JJA','SON'])
amps = np.array([0.7,1.4,2.1])

ymin = 100.
ymax = 1000.

pres = np.arange(100,1016,25)

mvcross    = Basemap()

for amps in [0.7, 1.4, 2.1]:
    fig,ax =plt.subplots()
    counter = np.zeros((len(pres),initdim))
    PVn = np.zeros(counter.shape)
    PRES = np.ones(PVn.shape)*pres[:,None]
    for simid,sim in enumerate(SIMS):
        if not str(amps) in sim:
            continue
        if 'DJF-nested' in sim or sim=='nested-test':
            continue
        if not 'DJF' in sim:
            continue

        tra = np.loadtxt(tracks + sim + '-01-new-tracks.txt')
        t = tra[:,0]
        tlon,tlat = tra[:,1],tra[:,2]
        slp = tra[:,3]
        IDs = tra[:,-1]

        loc = np.where(IDs==2)[0]
        lo = np.where(slp[loc] ==np.min(slp[loc]))[0][0]
        tm = t[loc[lo]]
        lonc,latc = tlon[loc[lo]],tlat[loc[lo]]
        dm =helper.simulation_time_to_day_string(tm)
        dlon = helper.convert_radial_distance_to_lon_lat_dis_new(dis,latc)

# Define cross section line for each date (tini)
        lon_start = lonc-dlon
        lon_end   = lonc+dlon
        lat_start = latc
        lat_end   = latc

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

        PVtmp =np.zeros((Var.shape[0],initdim))
        Prestmp = np.zeros((Var.shape[0],initdim))

        for k in range(Var.shape[0]):
            f_vcross     = Intergrid(Var[k,:,:], lo=bottomleft, hi=topright, verbose=0)
            f_p3d_vcross   = Intergrid(P[k,:,:], lo=bottomleft, hi=topright, verbose=0)
            for i in range(dimpath):
                vcross[k,i]     = f_vcross.at([latp[i],lonp[i]])
                vcross_p[k,i]   = f_p3d_vcross.at([latp[i],lonp[i]])


            PVtmp[k] = f_vcross.at(list(zip(latp,lonp)))
            Prestmp[k] = f_p3d_vcross.at(list(zip(latp,lonp)))

        maxpres = np.max(Prestmp,axis=0)
        prestmp=Prestmp
        for i in range(initdim):
            for l,u in enumerate(pres):
                    if u>maxpres[i]:
                        continue
                    else:
                        counter[l,i]+=1
                        PVn[l,i]+=PVtmp[np.where(abs(prestmp[:,i]-u)==np.min(abs(prestmp[:,i]-u)))[0][0],i]

    compovcross_PV = PVn/counter
    compovcross_p = PRES

    xcoord = np.zeros(shape=(PRES.shape[0],dimpath))
    for x in range(PRES.shape[0]):
       xcoord[x,:] = np.array([ i*deltas-dis for i in range(dimpath) ])


    cmap,pv_levels,norm,ticklabels=PV_cmap2()
    levels=pv_levels

    h = ax.contourf(xcoord, compovcross_p,
                        compovcross_PV,
                        levels = levels,
                        cmap = cmap,
                        norm=norm,extend='both')
    ax.set_xlabel('Distance to cyclone center [km]', fontsize=12)
    ax.set_ylabel('Pressure [hPa]', fontsize=12)
    ax.set_ylim(ymax,ymin)
    ax.set_yticks(np.arange(200,1001,100))
    ax.set_xlim(-500,500)
    ax.set_xticks(ticks=np.arange(-500,500,250))

    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(h, ticks=pv_levels,cax=cbax,extend='both')
    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
    fig.canvas.mpl_connect('draw_event', func)
    cbar.ax.set_xlabel('PVU',fontsize=10)
    figname = 'dom01-composite-vertical-cross-%s.png'%str(amps)
    fig.savefig(spaper+figname,dpi=300,bbox_inches='tight')
    plt.close('all')

plt.close('all')
