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

wrfd = '/atmosdyn2/ascherrmann/013-WRF-sim/'
for pvpo,sea in zip(pvpos,seasons):
    for amp in amps:

        sim=sea + '-clim-max-U-at-300-hPa-%.1f-QGPV'%amp
        ref = ds(wrfd + sea + '-clim/wrfout_d01_2000-12-01_00:00:00')
        ics = ds(wrfd + sim + '/wrfout_d01_2000-12-01_00:00:00')
        
        ### unperturbed data
        PVref=wrf.getvar(ref,'pvo')
        Pref = wrf.getvar(ref,'pressure')
        Uref = ref.variables['U'][0,:]
        MSLPref = ref.variables['MSLP'][0,:]
        PV300ref = wrf.interplevel(PVref,Pref,300,meta=False)

        ### perturbed data
        U = ics.variables['U'][0,:]
        PV = wrf.getvar(ics,'pvo')
        P = wrf.getvar(ics,'pressure')
        PV300 = wrf.interplevel(PV,P,300,meta=False)
        SLP = ics.variables['MSLP'][0,:]

        lon = wrf.getvar(ics,'lon')[0]
        lat = wrf.getvar(ics,'lat')[:,0]


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


        ### vertical perturbed
        Var=PV
        vcross_pv = np.zeros(shape=(Var.shape[0],dimpath))
        vcross_p= np.zeros(shape=(Var.shape[0],dimpath))
        ucross = np.zeros(shape=(Var.shape[0],dimpath))

        bottomleft = np.array([lat[0], lon[0]])
        topright   = np.array([lat[-1], lon[-1]])
        for k in range(Var.shape[0]):
            f_vcross     = Intergrid(Var[k,:,:], lo=bottomleft, hi=topright, verbose=0)
            f_p3d_vcross   = Intergrid(P[k,:,:], lo=bottomleft, hi=topright, verbose=0)
            f_ucross = Intergrid(U[k,:,:],lo=bottomleft, hi=topright, verbose=0)
            for i in range(dimpath):
                vcross_pv[k,i]     = f_vcross.at([latp[i],lonp[i]])
                vcross_p[k,i]   = f_p3d_vcross.at([latp[i],lonp[i]])
                ucross[k,i]    = f_ucross.at([latp[i],lonp[i]])


        ### figure
        fig=plt.figure(figsize=(15,5))
        gs = gridspec.GridSpec(nrows=2, ncols=4)

      
        ### ref PV cross 
        ax = fig.add_subplot(gs[0,:2],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')

        h2=ax.contour(lon,lat,MSLPref,levels=PSlevel,colors='purple',linewidths=0.5)
        hc=ax.contourf(lon,lat,PV300ref,cmap=pvcmap,norm=pvnorm,extend='both',levels=pvlevels)
        plt.clabel(h2,inline=True,fmt='%d',fontsize=6)

        ### pert PV cross
        ax2 = fig.add_subplot(gs[1,:2],projection=ccrs.PlateCarree())
        ax2.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')

        h2=ax2.contour(lon,lat,SLP,levels=PSlevel,colors='purple',linewidths=0.5)
        hc=ax2.contourf(lon,lat,PV300,cmap=pvcmap,norm=pvnorm,extend='both',levels=pvlevels)
        plt.clabel(h2,inline=True,fmt='%d',fontsize=6)

        ### vertical PV ref

        ax = fig.add_subplot(gs[0,2])
        xcoord = np.zeros(shape=(Var.shape[0],dimpath))
        for x in range(Var.shape[0]):
            xcoord[x,:] = np.array([ i*deltas-dis for i in range(dimpath) ])

        h = ax.contourf(xcoord, vcross_pref, vcross_pvref, levels = pvlevels, cmap = pvcmap, norm=pvnorm, extend = 'both')
        ax.set_ylabel('Pressure [hPa]', fontsize=12)
        ax.set_ylim(bottom=ymin, top=ymax)
        ax.set_xlim(-1 * dis,dis)
#        ax.set_xticks(ticks=np.arange(-1000,1000,250))
        ax.invert_yaxis()

        cbax = fig.add_axes([0, 0, 0.1, 0.1])
        cbar=plt.colorbar(h, ticks=pvlevels,cax=cbax)
        func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.005)
        fig.canvas.mpl_connect('draw_event', func)


       ### vertical PV pert

        ax = fig.add_subplot(gs[1,2])
        xcoord = np.zeros(shape=(Var.shape[0],dimpath))
        for x in range(Var.shape[0]):
            xcoord[x,:] = np.array([ i*deltas-dis for i in range(dimpath) ])

        h = ax.contourf(xcoord, vcross_p, vcross_pv, levels = pvlevels, cmap = pvcmap, norm=pvnorm, extend = 'both')
        ax.set_ylabel('Pressure [hPa]', fontsize=12)
        ax.set_ylim(bottom=ymin, top=ymax)
        ax.set_xlim(-1 * dis,dis)
#        ax.set_xticks(ticks=np.arange(-1000,1000,250))
        plt.gca().invert_yaxis()

        cbax = fig.add_axes([0, 0, 0.1, 0.1])
        cbar=plt.colorbar(h, ticks=pvlevels,cax=cbax)
        func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.005)
        fig.canvas.mpl_connect('draw_event', func)


        ### U ref vertical
        ax = fig.add_subplot(gs[0,3])
        h = ax.contourf(xcoord, vcross_p, urefcross,levels=ulevels,cmap=ucmap,norm=unorm,extend='max')
        #ax.set_ylabel('Pressure [hPa]', fontsize=12)
        ax.set_ylim(bottom=ymin, top=ymax)
        ax.set_xlim(-1 * dis,dis)
        ax.set_yticks([])
#        ax.set_xticks(ticks=np.arange(-1000,1000,250))
        plt.gca().invert_yaxis()

        cbax = fig.add_axes([0, 0, 0.1, 0.1])
        cbar=plt.colorbar(h, ticks=ulevels,cax=cbax)
        func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.005)
        fig.canvas.mpl_connect('draw_event', func)

        ### U vertical
        ax = fig.add_subplot(gs[1,3])
        ax.set_yticks([])
        h = ax.contourf(xcoord, vcross_p, ucross,levels=ulevels,cmap=ucmap,norm=unorm,extend='max')
        #ax.set_ylabel('Pressure [hPa]', fontsize=12)
        ax.set_ylim(bottom=ymin, top=ymax)
        ax.set_xlim(-1 * dis,dis)
#        ax.set_xticks(ticks=np.arange(-1000,1000,250))
        plt.gca().invert_yaxis()

        cbax = fig.add_axes([0, 0, 0.1, 0.1])
        cbar=plt.colorbar(h, ticks=ulevels,cax=cbax)
        func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.005)
        fig.canvas.mpl_connect('draw_event', func)
        plt.subplots_adjust(hspace=0.3)

        fig.savefig(wrfd + 'image-output/%s-PV-U-initial-overview.png'%sim,dpi=300,bbox_inches='tight')
        plt.close('all')
