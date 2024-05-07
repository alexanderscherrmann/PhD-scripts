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

SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/home/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

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

#for se in seas:
#  figg,axg = plt.subplots(figsize=(8,6))
#  scounter = 0 
#  for amp in amps:
#    fig,ax = plt.subplots(figsize=(8,6))
#    acounter = 0 
for simid,sim in enumerate(SIMS):
        if "-not" in sim:
            continue
        medid = np.array(MEDIDS[simid])
        atid = np.array(ATIDS[simid])

        if medid.size==0:
            continue

        sea = sim[:3]
#        if se!=sea:
#            continue

        tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
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
        if simid==18:
            svcross = np.zeros(shape=(Var.shape[0],dimpath))
            svcross_p= np.zeros(shape=(Var.shape[0],dimpath))
            avcross = np.zeros(shape=(Var.shape[0],dimpath))
            avcross_p= np.zeros(shape=(Var.shape[0],dimpath))
            sxcoord = np.zeros(shape=(Var.shape[0],dimpath))
            axcoord = np.zeros(shape=(Var.shape[0],dimpath))

        for k in range(Var.shape[0]):
            f_vcross     = Intergrid(Var[k,:,:], lo=bottomleft, hi=topright, verbose=0)
            f_p3d_vcross   = Intergrid(P[k,:,:], lo=bottomleft, hi=topright, verbose=0)
            for i in range(dimpath):
                vcross[k,i]     = f_vcross.at([latp[i],lonp[i]])
                vcross_p[k,i]   = f_p3d_vcross.at([latp[i],lonp[i]])

        fi,axx = plt.subplots(figsize=(8,6))

        xcoord = np.zeros(shape=(Var.shape[0],dimpath))
        for x in range(Var.shape[0]):
            xcoord[x,:] = np.array([ i*deltas-dis for i in range(dimpath) ])

        h = axx.contourf(xcoord, vcross_p, vcross, levels = levels, cmap = cmap, norm=norm, extend = 'both')

        axx.set_ylabel('Pressure [hPa]', fontsize=12)
        axx.set_ylim(bottom=ymin, top=ymax)

        axx.set_xlim(-1000,1000)
        axx.set_xticks(ticks=np.arange(-1000,1000,250))
        axx.invert_yaxis()

        cbax = fi.add_axes([0, 0, 0.1, 0.1])
        cbar=plt.colorbar(h, ticks=levels,cax=cbax)
        func=resize_colorbar_vert(cbax, axx, pad=0.0, size=0.02)
        fi.canvas.mpl_connect('draw_event', func)

        fi.savefig(dwrf + 'image-output/vertical-cross-sims/%s-vertical-cross.png'%sim,dpi=300,bbox_inches='tight')
        plt.close(fi)

        #print(svcross.shape,vcross.shape)
        #try:
        #    svcross += vcross
        #    svcross_p += vcross_p
        #except:
        #    svcross[:,:vcross.shape[1]-svcross.shape[1]] +=vcross
        #    svcross_p[:,:vcross.shape[1]-svcross.shape[1]] +=vcross_p

        #scounter==1

        #if float(sim[-8:-5])==amp:
        #    acounter+=1
        #    try:
        #        avcross += vcross
        #        avcross_p += vcross_p
        #    except:
        #        avcross[:,:vcross.shape[1]-avcross.shape[1]] +=vcross
        #        avcross_p[:,:vcross.shape[1]-avcross.shape[1]] +=vcross_p

    #h = ax.contourf(xcoord, avcross_p/acounter, avcross/acounter, levels = levels, cmap = cmap, norm=norm, extend = 'both')
    #ax.set_ylabel('Pressure [hPa]', fontsize=12)

    #ax.set_xlim(-1000,1000)
    #ax.set_xticks(ticks=np.arange(-1000,1000,250))
    #ax.invert_yaxis()
    #ax.set_ylim(bottom=ymin, top=ymax)
    #cbax = fig.add_axes([0, 0, 0.1, 0.1])
    #cbar=plt.colorbar(h, ticks=levels,cax=cbax)
    #func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
    #fig.canvas.mpl_connect('draw_event', func)

    #fig.savefig(dwrf + 'image-output/vertical-cross-sims/%s-%.1f-vertical-cross.png'%(se,amp),dpi=300,bbox_inches='tight')
    #plt.close(fig)

  #h = axg.contourf(xcoord, avcross_p/scounter, avcross/scounter, levels = levels, cmap = cmap, norm=norm, extend = 'both')
  #axg.set_ylabel('Pressure [hPa]', fontsize=12)
  #axg.set_ylim(bottom=ymin, top=ymax)

  #axg.set_xlim(-1000,1000)
  #axg.set_xticks(ticks=np.arange(-1000,1000,250))
  #axg.invert_yaxis()

  #cbax = figg.add_axes([0, 0, 0.1, 0.1])
  #cbar=plt.colorbar(h, ticks=levels,cax=cbax)
  #func=resize_colorbar_vert(cbax, axg, pad=0.0, size=0.02)
  #figg.canvas.mpl_connect('draw_event', func)

  #figg.savefig(dwrf + 'image-output/vertical-cross-sims/%s-vertical-cross.png'%(se),dpi=300,bbox_inches='tight')
  #plt.close(figg)

plt.close('all')
