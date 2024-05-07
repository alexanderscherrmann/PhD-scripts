import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wrf import interplevel as intp
from netCDF4 import Dataset as ds
from mpl_toolkits.basemap import Basemap
from dypy.intergrid import Intergrid
from matplotlib.colors import ListedColormap, BoundaryNorm
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
import helper
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
era5 = '/atmosdyn2/era5/cdf/'

which = ['weak-cyclones.csv','intense-cyclones.csv']

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

dis = 500 # longitudinal distance to cyclone center to west and east
dds = 5.

mvcross    = Basemap()
initl, = mvcross.drawgreatcircle(-1 * helper.convert_radial_distance_to_lon_lat_dis_new(dis,0),0,helper.convert_radial_distance_to_lon_lat_dis_new(dis,0),0,del_s=dds)
initpa = initl.get_path()
initlop,initlap = mvcross(initpa.vertices[:,0],initpa.vertices[:,1], inverse=True)
initdim = len(initlop)

bottomleft = np.array([-90., -180.])
topright   = np.array([90., 179.5])

# average cyclone 
we = 'dates'

P = ds(era5 + '2000/10/P20001010_10',mode='r')
hyam=P.variables['hyam']  # 137 levels  #für G-file ohne levels bis
hybm=P.variables['hybm']  #   ''
ak=hyam[hyam.shape[0]-98:] # only 98 levs are used:
bk=hybm[hybm.shape[0]-98:]

pres = np.arange(200,1016,25)

#                          -0.5-0    0.25           0.5          0.75      1.0          1.25         1.5     2.

levels = np.array([-0.5,0,0.25,0.5,0.75,1.0,1.25,1.5,2,3.,4.,5.,6.,7.,8.,9.,10.,12])
cmap = ListedColormap(np.array(['navy','dodgerblue','lightskyblue','salmon','lightcoral','indianred','crimson','red','darkorange','orange','gold','yellow','yellowgreen','limegreen','olivedrab','forestgreen']))
norm = BoundaryNorm(levels,cmap.N)
ticklabels=levels

seasons = ['DJF']#,'MAM','JJA','SON']
for sea in seasons:
    for wi in which:
      sel = pd.read_csv(ps + sea + '-' + wi)
      for ll in [50, 100, 150, 200][-1:]:
        selp = sel.iloc[:ll]
        counter = np.zeros((len(pres),initdim))
        PVn = np.zeros(counter.shape)
        PRES = np.ones(PVn.shape)*pres[:,None]
        THn = np.zeros(counter.shape)

        for d,lonc,latc in zip(selp[we].values,selp['lon'].values,selp['lat'].values):
            dlon = helper.convert_radial_distance_to_lon_lat_dis_new(dis,latc)

            los = lonc-dlon
            loe = lonc+dlon

            ep = era5 + d[:4] + '/' + d[4:6] + '/' 
            S = ds(ep + 'S' + d,mode='r')
            P = ds(ep + 'P' + d,mode='r')
            PV = S.variables['PV'][0]
            PS = S.variables['PS'][0]
            TH = S.variables['TH'][0]
            
            ps3d=np.tile(PS[:,:],(len(ak),1,1))
            p3d=(ak/100.+bk*ps3d.T).T

            line,= mvcross.drawgreatcircle(los, latc, loe, latc, del_s=dds)
            path       = line.get_path()
            lonp, latp = mvcross(path.vertices[:,0], path.vertices[:,1], inverse=True)
            dimpath    = len(lonp)

            PVtmp = np.zeros((len(ak),initdim))
            prestmp = np.zeros(PVtmp.shape)
            THtmp = np.zeros((len(ak),initdim))
            for k in range(len(ak)):
                f_vcross_PV   = Intergrid(PV[k,:,:], lo=bottomleft, hi=topright, verbose=0)
                f_vcross_TH = Intergrid(TH[k,:,:], lo=bottomleft, hi=topright, verbose=0)
                f_p3d_vcross   = Intergrid(p3d[k,:,:], lo=bottomleft, hi=topright, verbose=0)
        
                PVtmp[k] = f_vcross_PV.at(list(zip(latp,lonp)))
                prestmp[k] = f_p3d_vcross.at(list(zip(latp,lonp)))
                THtmp[k] = f_vcross_TH.at(list(zip(latp,lonp)))

            maxpres = np.max(prestmp,axis=0)
            for i in range(initdim):
                for l,u in enumerate(pres):
                    if u>maxpres[i]:
                        continue
                    else:
                        counter[l,i]+=1
                        PVn[l,i]+=PVtmp[np.where(abs(prestmp[:,i]-u)==np.min(abs(prestmp[:,i]-u)))[0][0],i]
                        THn[l,i]+=THtmp[np.where(abs(prestmp[:,i]-u)==np.min(abs(prestmp[:,i]-u)))[0][0],i]

        compovcross_TH = THn/counter
        compovcross_PV = PVn/counter
        compovcross_p = PRES

        xcoord = np.zeros(shape=(len(pres),dimpath))
        for x in range(len(pres)):
            xcoord[x,:] = np.array([ i*dds-dis for i in range(dimpath) ])

        levels=levels
        ymin = 200.
        ymax = 1000.
        unit = 'PVU'
        

        fig, ax = plt.subplots()
        h = ax.contourf(xcoord, compovcross_p,
                        compovcross_PV,
                        levels = levels,
                        cmap = cmap,
                        norm=norm,extend='both')

        th_spec=np.linspace(250,420, num=35)
        cs = ax.contour(xcoord,compovcross_p,compovcross_TH,levels=th_spec,colors='#606060',linewidths=1)
        plt.clabel(cs, inline=1, fontsize=8, fmt='%d')
        
        ax.set_xlabel('Distance to cyclone center [km]', fontsize=12)
        ax.set_ylabel('Pressure [hPa]', fontsize=12)
        ax.set_ylim(ymax,ymin)
        ax.set_xlim(-500,500)
        ax.set_xticks(ticks=np.arange(-500,500,250))
        
        cbax = fig.add_axes([0, 0, 0.1, 0.1])
        cbar=plt.colorbar(h, ticks=levels,cax=cbax,extend='both')
        func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
        fig.canvas.mpl_connect('draw_event', func)
        cbar.ax.set_xlabel('PVU',fontsize=10)
        
        name = 'composite-PV-crosssection-' + wi[:-4] + '-%d-'%ll + sea + '.png'
        fig.savefig(pi + name,dpi=300,bbox_inches='tight')
        plt.close('all')


