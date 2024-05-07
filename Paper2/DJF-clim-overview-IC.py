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
pappath = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'
for pvpo,sea in zip(pvpos,seasons[:1]):
#    for amp in amps:

#        sim=sea + '-clim-max-U-at-300-hPa-%.1f-QGPV'%amp
        ref = ds(wrfd + sea + '-clim/wrfout_d01_2000-12-01_00:00:00')
#        ics = ds(wrfd + sim + '/wrfout_d01_2000-12-01_00:00:00')
        
        ### unperturbed data
        PVref=wrf.getvar(ref,'pvo')
        Pref = wrf.getvar(ref,'pressure')
        Uref = ref.variables['U'][0,:]
        print(Uref.shape,Pref.shape)
        MSLPref = ref.variables['MSLP'][0,:]
        PV300ref = wrf.interplevel(PVref,Pref,300,meta=False)
        U300ref = wrf.interplevel((Uref[:,:,:-1] + Uref[:,:,1:])/2,Pref,300,meta=False)

        ### perturbed data
#        U = ics.variables['U'][0,:]
#        PV = wrf.getvar(ics,'pvo')
#        P = wrf.getvar(ics,'pressure')
#        PV300 = wrf.interplevel(PV,P,300,meta=False)
#        SLP = ics.variables['MSLP'][0,:]

        lon = wrf.getvar(ref,'lon')[0]
        lat = wrf.getvar(ref,'lat')[:,0]


        ### vertical ref
        lon_start = lon[pvpo[0]]
        lon_end=lon_start
        lat_start = lat[pvpo[1]] - dlat
        lat_end = lat_start+2*dlat

        print(lon_start, lat_start, lon_end, lat_end)

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
#        Var=PV
#        vcross_pv = np.zeros(shape=(Var.shape[0],dimpath))
#        vcross_p= np.zeros(shape=(Var.shape[0],dimpath))
#        ucross = np.zeros(shape=(Var.shape[0],dimpath))
#
#        bottomleft = np.array([lat[0], lon[0]])
#        topright   = np.array([lat[-1], lon[-1]])
#        for k in range(Var.shape[0]):
#            f_vcross     = Intergrid(Var[k,:,:], lo=bottomleft, hi=topright, verbose=0)
#            f_p3d_vcross   = Intergrid(P[k,:,:], lo=bottomleft, hi=topright, verbose=0)
#            f_ucross = Intergrid(U[k,:,:],lo=bottomleft, hi=topright, verbose=0)
#            for i in range(dimpath):
#                vcross_pv[k,i]     = f_vcross.at([latp[i],lonp[i]])
#                vcross_p[k,i]   = f_p3d_vcross.at([latp[i],lonp[i]])
#                ucross[k,i]    = f_ucross.at([latp[i],lonp[i]])


        ### figure
        fig=plt.figure(figsize=(16,5))
        gs = gridspec.GridSpec(nrows=1, ncols=5)

        ### ref PV cross 
        ax = fig.add_subplot(gs[0,:2],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')

        h2=ax.contour(lon,lat,MSLPref,levels=PSlevel,colors='purple',linewidths=0.5)
        hc=ax.contourf(lon,lat,PV300ref,cmap=pvcmap,norm=pvnorm,extend='both',levels=pvlevels)
        plt.clabel(h2,inline=True,fmt='%d',fontsize=6)
        cbax = fig.add_axes([0, 0, 0.1, 0.1])
        cbar=plt.colorbar(hc, ticks=pvlevels,cax=cbax)
        func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.005)
        fig.canvas.mpl_connect('draw_event', func)
        
        ax.set_aspect('auto')

        ### pert PV cross
        ax2=fig.add_subplot(gs[0,2:4],projection=ccrs.PlateCarree())
        ax2.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
        ax2.contour(lon,lat,PV300ref,levels=[2],colors='k',linewidths=2)
        hc=ax2.contourf(lon,lat,U300ref,cmap=ucmap,norm=unorm,extend='max',levels=ulevels)

        ax2.plot([lon_start, lon_end], [lat_start, lat_end],color='grey',linewidth=2)
        cbax = fig.add_axes([0, 0, 0.1, 0.1])
        cbar=plt.colorbar(hc, ticks=ulevels,cax=cbax)
        func=resize_colorbar_vert(cbax, ax2, pad=0.0, size=0.005)
        fig.canvas.mpl_connect('draw_event', func)
        ax2.set_aspect('auto')
        ### vertical U ref

        ax = fig.add_subplot(gs[0,4])
        xcoord = np.zeros(shape=(Var.shape[0],dimpath))
        for x in range(Var.shape[0]):
            xcoord[x,:] = np.array([ i*deltas-dis for i in range(dimpath) ])

        h = ax.contourf(xcoord, vcross_pref, urefcross, levels = ulevels, cmap =ucmap, norm=unorm, extend = 'max')
        ax.set_ylabel('Pressure [hPa]',fontsize=8)
        ax.set_ylim(bottom=ymin, top=ymax)
        ax.set_xlim(-1 * dis,dis)
        ax.set_xticks(ticks=np.arange(-1000,1100,500))
        ax.set_xticklabels(np.arange(-1000,1100,500),fontsize=8)
        ax.set_yticklabels(np.arange(0,1200,200),fontsize=8)
        ax.invert_yaxis()

        cbax = fig.add_axes([0, 0, 0.1, 0.1])
        cbar=plt.colorbar(h, ticks=ulevels,cax=cbax)
        func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.005)
        fig.canvas.mpl_connect('draw_event', func)

        ax.set_aspect('auto')
        
        plt.subplots_adjust(top=0.45,wspace=0.4)
        pos2 = ax2.get_position()
        pos2n = [pos2.x0 - 0.01,pos2.y0,pos2.width,pos2.height]
        ax2.set_position(pos2n)


        print(fig.axes)
#        plt.tight_layout()
        fig.savefig(pappath + '%s-clim-IC-overview.png'%sea,dpi=300,bbox_inches='tight')
        plt.close('all')
