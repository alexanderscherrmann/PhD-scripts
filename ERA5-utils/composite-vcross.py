def readcdf(ncfile,varnam):
    infile = netCDF4.Dataset(ncfile, mode='r')
    var = infile.variables[varnam][:]
    return(var)

def inter2level(varr3D, parr3D, plevel):
    """
    Interpolates 3-D (level, lat, lon) over level for variable array varr with
    associated pressure grid parr to the scalar pressure level plevel
    """
    v_i = interpolate(varr3D[::1,:, :], parr3D[:, :], plevel)
    return(v_i)


from dypy.intergrid import Intergrid
import matplotlib.colors as col
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
import numpy as np
import netCDF4
from netCDF4 import Dataset as ncFile
from dypy.small_tools import interpolate
from dypy.lagranto import Tra
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec
from matplotlib.colors import from_levels_and_colors
import datetime as dt
from matplotlib.colors import BoundaryNorm,ListedColormap
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
import helper
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import argparse 
import pandas as pd
import pickle
### load cyclone data
###

parser = argparse.ArgumentParser(description="composite vertical cross section of XX ocean below XXX hPa")
#parser.add_argument('ocean', default='MED', type=str, help='ocean: MED, NA, SA, NP, SP, IO')
#parser.add_argument('slpb', default=1, type=int, help='composite for all cyclones below XXX hPa')
args = parser.parse_args()

pload = '/home/ascherrmann/009-ERA-5/MED/traj/'

DF = pd.read_csv(pload + 'pandas-all-data.csv')
#slpb = int(args.slpb)
#reg = str(args.ocean)
ntraj = 2000
dates = DF.loc[(DF['ntraj075']>=ntraj)]['date'].values
lons = DF.loc[(DF['ntraj075']>=ntraj)]['lon'].values
lats = DF.loc[(DF['ntraj075']>=ntraj)]['lat'].values

#IDs = DF.loc[(DF['reg']=='MED') & (DF['minSLP']<1000)].index.values


#dates = DF.loc[(DF['reg']==reg)]['dates'].values
#lons = DF.loc[(DF['reg']==reg)]['lon'].values
#lats = DF.loc[(DF['reg']==reg)]['lat'].values


#fig = plt.figure(figsize=(8,6))
#gs = gridspec.GridSpec(ncols=1, nrows=1)
#ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
#ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
#ax.scatter(lons,lats)
#minpltlonc = -5
#minpltlatc = 25
#maxpltlonc=45
#maxpltlatc = 50
#ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
#for ID, lon,lat in zip(IDs,lons,lats):
#    ax.annotate('%d'%ID,(lon,lat))
#fig.savefig('/home/ascherrmann/011-all-ERA5/MEDtest.png',dpi=300,bbox_inches="tight")
#plt.close('all')


#bo = 246
#ad = 5
#dates = dates[bo:bo+ad]
#lons = lons[bo:bo+ad]
#lats = lats[bo:bo+ad]

### vars needed in calculation
###
dis = 500 # longitudinal distance to cyclone center to west and east
ds = 5. ## distance for crosssection

### prepare array for  average
###
mvcross    = Basemap()
initl, = mvcross.drawgreatcircle(-1 * helper.convert_radial_distance_to_lon_lat_dis_new(dis,0),0,helper.convert_radial_distance_to_lon_lat_dis_new(dis,0),0,del_s=ds)
initpa = initl.get_path()
initlop,initlap = mvcross(initpa.vertices[:,0],initpa.vertices[:,1], inverse=True)
initdim = len(initlop)

bottomleft = np.array([-90., -180.])
topright   = np.array([90., 179.5])


### load modellevel data
pf = '/atmosdyn/era5/cdf/2018/09/P20180928_04'

sav=30
hyam=readcdf(pf,'hyam')  # 137 levels  #für G-file ohne levels bis
hybm=readcdf(pf,'hybm')  #   ''
ak=hyam[hyam.shape[0]-98 + sav:] # only 98 levs are used:
bk=hybm[hybm.shape[0]-98 + sav:] #remove calculation above level of interest ~ 160 hPa

### define background pressure grid
###
pres = np.arange(200,1016,25)

###craete average arrays
###
counter = np.zeros((len(pres),initdim))
PVn = np.zeros(counter.shape)
PRES = np.ones(PVn.shape)*pres[:,None]
#compovcross_PV = np.zeros(shape=(len(ak),initdim))
#compovcross_p = np.zeros(shape=(len(ak),initdim))


### verification of calculation for zorbas
#dates = ['20180928_04']
#lons = [19]
#lats =[34.5]
####



### start calculation
###
for dat,lonc,latc in zip(dates,lons,lats):
    dlon = helper.convert_radial_distance_to_lon_lat_dis_new(dis,latc)
    los = lonc-dlon
    loe = lonc+dlon

    y=dat[0:4]
    m=dat[4:6]

    sfile='/net/thermo/atmosdyn/era5/cdf/'+y+'/'+m+'/S'+dat

    ps=readcdf(sfile,'PS')
    PV=readcdf(sfile,'PV') 
    PV = PV[0,sav:]
    line,= mvcross.drawgreatcircle(los, latc, loe, latc, del_s=ds)
    path       = line.get_path()
    lonp, latp = mvcross(path.vertices[:,0], path.vertices[:,1], inverse=True)
    dimpath    = len(lonp)

    p3d=np.full((len(ak),PV.shape[0],PV.shape[1]),-999.99)
    ps3d=np.tile(ps[0,:,:],(len(ak),1,1)) # write/repete ps to each level of dim 0
    p3d=(ak/100.+bk*ps3d.T).T

#    vcross_PV = np.zeros(shape=(PV.shape[1],dimpath))
#    vcross_p  = np.zeros(shape=(p3d.shape[0],dimpath)) #pressure
    PVtmp = np.zeros((len(ak),initdim))
    prestmp = np.zeros(PVtmp.shape)
    for k in range(len(ak)):
        f_vcross_PV   = Intergrid(PV[k,:,:], lo=bottomleft, hi=topright, verbose=0)
        f_p3d_vcross   = Intergrid(p3d[k,:,:], lo=bottomleft, hi=topright, verbose=0)

        PVtmp[k] = f_vcross_PV.at(list(zip(latp,lonp)))
        prestmp[k] = f_p3d_vcross.at(list(zip(latp,lonp)))

    maxpres = np.max(prestmp,axis=0)
    for i in range(initdim):
        for l,u in enumerate(pres):
            if u>maxpres[i]:
                continue
            else:
                counter[l,i]+=1
                PVn[l,i]+=PVtmp[np.where(abs(prestmp[:,i]-u)==np.min(abs(prestmp[:,i]-u)))[0][0],i]

compovcross_PV = PVn/counter
compovcross_p = PRES

### Create coorinate array for x-axis
###
xcoord = np.zeros(shape=(len(pres),dimpath))
for x in range(len(pres)):
    xcoord[x,:] = np.array([ i*ds-dis for i in range(dimpath) ])

### plotting
###

cmap,pv_levels,norm,ticklabels=PV_cmap2()

levels=pv_levels
outpath  = '/home/ascherrmann/009-ERA-5/MED/'
ymin = 200.
ymax = 1000.
unit = 'PVU'


fig, ax = plt.subplots()
h = ax.contourf(xcoord, compovcross_p,
                compovcross_PV,
                levels = levels,
                cmap = cmap,
                norm=norm,extend='both')

ax.set_xlabel('Distance to cyclone center [km]', fontsize=12)
ax.set_ylabel('Pressure [hPa]', fontsize=12)
ax.set_ylim(ymax,ymin)
ax.set_xlim(-500,500)
ax.set_xticks(ticks=np.arange(-500,500,250))


cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(h, ticks=pv_levels,cax=cbax,extend='both')
func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
fig.canvas.mpl_connect('draw_event', func)
cbar.ax.set_xlabel('PVU',fontsize=10)

figname = 'composite-PV-crossection-ERA5-2000traj.png'
fig.savefig(outpath+figname, dpi=300,bbox_inches = 'tight')
plt.close(fig)


