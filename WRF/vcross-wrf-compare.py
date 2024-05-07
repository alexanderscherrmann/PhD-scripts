from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, BoundaryNorm
from wrf import (getvar, to_np, vertcross, smooth2d, CoordPair,
                 get_basemap, latlon_coords)
import wrf
import argparse
def readcdf(ncfile,varnam):
    infile = netCDF4.Dataset(ncfile, mode='r')
    var = infile.variables[varnam][:]
    return(var)
    
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
import matplotlib
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.colors import from_levels_and_colors

import datetime as dt
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
import helper
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert

parser = argparse.ArgumentParser(description="composite vertical cross section of XX ocean below XXX hPa")
parser.add_argument('day',default='',type=str,help='folder/simulation for which to evaluate surface pressure and PV at 300 hPa')
parser.add_argument('lon',default='',type=float,help='folder/simulation for which to evaluate surface pressure and PV at 300 hPa')

args = parser.parse_args()
d=str(args.day)
lo = float(args.lon)

ymin = 200.
ymax = 1000.

lon_start = lo
lon_end   = lo
lat_start = 10
lat_end   = 80

dis = (lat_end-lat_start)/360 * 2 * np.pi * 6370

### date
path = '/atmosdyn/era5/cdf/' + d[:4] + '/' + d[4:6] +'/'
pfile = path +'P' + d
sfile = path +'S' + d
### vars
ps=readcdf(pfile,'PS')
T=readcdf(pfile,'T')
U=readcdf(pfile,'U')
V=readcdf(pfile,'V')
PV=readcdf(sfile,'PV')
TH=readcdf(sfile,'TH')
pv=PV


lons=readcdf(pfile,'lon')
lats=readcdf(pfile,'lat')
hyam=readcdf(pfile,'hyam')  # 137 levels  #für G-file ohne levels bis
hybm=readcdf(pfile,'hybm')  #   ''
ak=hyam[hyam.shape[0]-pv.shape[1]:] # only 98 levs are used:
bk=hybm[hybm.shape[0]-pv.shape[1]:]

### line on cross section
ds = 5.
mvcross    = Basemap()
line,      = mvcross.drawgreatcircle(lon_start, lat_start, lon_end, lat_end, del_s=ds)
path       = line.get_path()
lonp, latp = mvcross(path.vertices[:,0], path.vertices[:,1], inverse=True)
dimpath    = len(lonp)


# calculate pressure on model levels
p3d=np.full((pv.shape[1],pv.shape[2],pv.shape[3]),-999.99)
ps3d=np.tile(ps[0,:,:],(pv.shape[1],1,1)) # write/repete ps to each level of dim 0
p3d=(ak/100.+bk*ps3d.T).T
unit_p3d = 'hPa'


### allocate cross section
vcross = np.zeros(shape=(pv.shape[1],dimpath)) #PV
vcross_p  = np.zeros(shape=(p3d.shape[0],dimpath)) #pressure
vcross_TH = np.zeros(shape=(TH.shape[1],dimpath))

bottomleft = np.array([lats[0], lons[0]])
topright   = np.array([lats[-1], lons[-1]])


for k in range(pv.shape[1]):
    f_vcross     = Intergrid(pv[0,k,:,:], lo=bottomleft, hi=topright, verbose=0)
    f_vcross_TH   = Intergrid(TH[0,k,:,:], lo=bottomleft, hi=topright, verbose=0)
    f_p3d_vcross   = Intergrid(p3d[k,:,:], lo=bottomleft, hi=topright, verbose=0)
    for i in range(dimpath):
        vcross[k,i]     = f_vcross.at([latp[i],lonp[i]])
        vcross_TH[k,i]   = f_vcross_TH.at([latp[i],lonp[i]])
        vcross_p[k,i]   = f_p3d_vcross.at([latp[i],lonp[i]])
        
xcoord = np.zeros(shape=(pv.shape[1],dimpath))
for x in range(pv.shape[1]):
    xcoord[x,:] = np.array([ i*ds-dis for i in range(dimpath) ])

###
### WRF data
###

data = netCDF4.Dataset('/home/ascherrmann/scripts/WRF/wrf-mean-reference')
PVwrf = wrf.getvar(data,'pvo',meta=False) #in PVU already
lonwrf = wrf.getvar(data,'lon',meta=False)[0]-0.25
latwrf = wrf.getvar(data,'lat',meta=False)[:,0]-0.25

THwrf = wrf.getvar(data,'T',meta=False) + 300
Pwrf = wrf.getvar(data,'pressure',meta=False)


###
### do line for cross section for wrf
###

linew,      = mvcross.drawgreatcircle(lon_start, lat_start, lon_end, lat_end, del_s=ds)
pathw       = line.get_path()
lonpw, latpw = mvcross(pathw.vertices[:,0], pathw.vertices[:,1], inverse=True)
dimpathw    = len(lonpw)
p3dw = Pwrf

bottomleftw = np.array([latwrf[0], lonwrf[0]])
toprightw   = np.array([latwrf[-1], lonwrf[-1]])

vcrossw = np.zeros(shape=(Pwrf.shape[0],dimpathw))
vcross_pw = np.zeros(shape=(Pwrf.shape[0],dimpathw))
vcross_THw = np.zeros(shape=(Pwrf.shape[0],dimpathw))

for k in range(Pwrf.shape[0]):
    f_vcrossw     = Intergrid(PVwrf[k,:,:], lo=bottomleftw, hi=toprightw, verbose=0)
    f_TH_vcrossw = Intergrid(THwrf[k,:,:], lo=bottomleftw, hi=toprightw, verbose=0)
    f_p3d_vcrossw = Intergrid(p3dw[k,:,:], lo=bottomleftw, hi=toprightw, verbose=0)
    for i in range(dimpath):
        vcrossw[k,i]     = f_vcrossw.at([latpw[i],lonpw[i]])
        vcross_pw[k,i]   =f_p3d_vcrossw.at([latpw[i],lonpw[i]])
        vcross_THw[k,i] = f_TH_vcrossw.at([latpw[i],lonpw[i]])


xcoordw = np.zeros(shape=(Pwrf.shape[0],dimpath))
for x in range(Pwrf.shape[0]):
    xcoordw[x,:] = np.array([ i*ds-dis for i in range(dimpath) ])

#
# create figure
#

fig,axes = plt.subplots(1,2,sharey=True)
plt.subplots_adjust(wspace=0.25,top=0.5)
axes=axes.flatten()

PVlvl = np.arange(-2,2.1,0.25)
THlvl = np.arange(-10,10.1,2)

cmap = matplotlib.cm.BrBG
norm = plt.Normalize(np.min(PVlvl),np.max(PVlvl))

ids = np.array([])
for p in np.mean(p3dw,axis=(1,2)):
    ids = np.append(ids,np.where(abs(np.mean(p3d,axis=(1,2))-p)== np.min(abs(np.mean(p3d,axis=(1,2))-p)))[0][0])

ids = ids.astype(int)

h=axes[0].contourf(xcoordw,vcross_pw,vcrossw-vcross[ids],levels=PVlvl,cmap=cmap,norm=norm,extend='both')

unit='PVU'
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(h, ticks=PVlvl[::2],cax=cbax)
func=resize_colorbar_vert(cbax, axes[0], pad=0.0, size=0.02)
fig.canvas.mpl_connect('draw_event', func)
cbar.ax.set_xlabel(unit)

cmap = matplotlib.cm.seismic
norm = plt.Normalize(np.min(THlvl),np.max(THlvl))

h = axes[1].contourf(xcoordw,vcross_pw,vcross_THw-vcross_TH[ids],levels=THlvl,cmap=cmap,norm=norm,extend='both')

axes[0].set_ylabel('Pressure [hPa]', fontsize=12)
axes[0].set_ylim(bottom=ymin, top=ymax)
axes[0].invert_yaxis()

unit='K'
cbax = fig.add_axes([1.0, 1.0, 0.1, 0.1])
cbar=plt.colorbar(h, ticks=THlvl[::2],cax=cbax)
func=resize_colorbar_vert(cbax, axes[1], pad=0.0, size=0.02)
fig.canvas.mpl_connect('draw_event', func)
cbar.ax.set_xlabel(unit)

figname = '/atmosdyn2/ascherrmann/012-WRF-cyclones/' + d + '-%d-PV-TH-comp.png'%int(lo)
fig.savefig(figname, bbox_inches = 'tight',dpi=300)
plt.close(fig)


