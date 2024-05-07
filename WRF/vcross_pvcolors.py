#!/usr/bin/env python

#%%############ functions ####################
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


############# end functions ##################   


## Import modules
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
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("file",default='',type=str,help='path to file')
args = parser.parse_args()
pfile = str(args.file)


# Define cross section line for each date (tini)
lon_start = -65
lon_end   = -65
lat_start = 10
lat_end   = 80

dis = (lat_end-lat_start)/360 * 2 * np.pi * 6370

# Define variable
var='T'
unit='m/s'#g kg$^{-1}$'

cmap,pv_levels,norm,ticklabels=PV_cmap2()
levels=pv_levels
# Define lower and upper bound for vertical cross section (y-axis)
ymin = 50.
ymax = 1000.

dynfmt = '%Y%m%d_%H'
#datobj=dt.datetime.strptime(dat,dynfmt)  # change to python time
outpath  = './'#atmosdyn2/ascherrmann/009-ERA-5/MED/' #Wo Plot gespeichert wird

#pfile='Pmean'#'/net/thermo/atmosdyn/era5/cdf/'+y+'/'+m+'/P'+dat  #für Lothar ändern clim_era5/lothar ohne +y+ und +m+
ps=readcdf(pfile,'PS')
T=readcdf(pfile,'T')
U=readcdf(pfile,'U')
V=readcdf(pfile,'V')
#PV=readcdf('S' + pfile[1:],'PV')
Z=readcdf('H' + pfile[1:],'Z')
Z/=9.81
pv=T
#TH=readcdf('S' + pfile[1:],'TH')

#q=readcdf(pfile,'Q')
#rh=readcdf(sfile,'RH')
#th=readcdf(sfile,'TH')
#the=readcdf(sfile,'THE')
lons=readcdf(pfile,'lon')
lats=readcdf(pfile,'lat')
hyam=readcdf(pfile,'hyam')  # 137 levels  #für G-file ohne levels bis
hybm=readcdf(pfile,'hybm')  #   ''
ak=hyam[hyam.shape[0]-pv.shape[1]:] # only 98 levs are used:
bk=hybm[hybm.shape[0]-pv.shape[1]:] # reduce to 98 levels 

# Define distance delta for great circle line
ds = 5.

# Extract coordinates of great circle line between start and end point
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

# Extract data along the great circle line between the start and end point
vcross = np.zeros(shape=(pv.shape[1],dimpath)) #PV
vcross_T = np.zeros(shape=(T.shape[1],dimpath))
vcross_U= np.zeros(shape=(T.shape[1],dimpath))
vcross_V= np.zeros(shape=(T.shape[1],dimpath))
vcross_p  = np.zeros(shape=(p3d.shape[0],dimpath)) #pressure
#vcross_PV = np.zeros(shape=(T.shape[1],dimpath))
#vcross_TH = np.zeros(shape=(T.shape[1],dimpath))

bottomleft = np.array([lats[0], lons[0]])
topright   = np.array([lats[-1], lons[-1]])

vcross_Z = np.zeros(shape=(Z.shape[1],dimpath))
vcross_pZ = np.zeros(shape=(Z.shape[1],dimpath)) 
pH = readcdf('H' + pfile[1:],'plev')

for k in range(pv.shape[1]):
    f_vcross     = Intergrid(pv[0,k,:,:], lo=bottomleft, hi=topright, verbose=0)
    f_vcross_T   = Intergrid(T[0,k,:,:], lo=bottomleft, hi=topright, verbose=0)
    f_vcross_U = Intergrid(U[0,k,:,:], lo=bottomleft, hi=topright, verbose=0)
    f_vcross_V = Intergrid(V[0,k,:,:], lo=bottomleft, hi=topright, verbose=0)
#    f_vcross_PV = Intergrid(PV[0,k,:,:], lo=bottomleft, hi=topright, verbose=0)
#    f_vcross_TH = Intergrid(TH[0,k,:,:], lo=bottomleft, hi=topright, verbose=0)
    f_p3d_vcross   = Intergrid(p3d[k,:,:], lo=bottomleft, hi=topright, verbose=0)
    for i in range(dimpath):
        vcross[k,i]     = f_vcross.at([latp[i],lonp[i]])
        vcross_T[k,i]   = f_vcross_T.at([latp[i],lonp[i]])
        vcross_p[k,i]   = f_p3d_vcross.at([latp[i],lonp[i]])
        vcross_U[k,i]   = f_vcross_U.at([latp[i],lonp[i]])
        vcross_V[k,i]   = f_vcross_V.at([latp[i],lonp[i]])
#        vcross_PV[k,i] = f_vcross_PV.at([latp[i],lonp[i]])
#        vcross_TH[k,i] = f_vcross_TH.at([latp[i],lonp[i]])

for k in range(Z.shape[1]):
    f_vcross_Z = Intergrid(Z[0,k,:,:],lo=bottomleft, hi=topright, verbose=0)

    for i in range(dimpath):
        vcross_Z[k,i] =f_vcross_Z.at([latp[i],lonp[i]])
        vcross_pZ[k,i] = pH[k]



    
# Create coorinate array for x-axis
xcoord = np.zeros(shape=(pv.shape[1],dimpath))
for x in range(pv.shape[1]):
    xcoord[x,:] = np.array([ i*ds-dis for i in range(dimpath) ])

xcoordZ = np.zeros(shape=(Z.shape[1],dimpath))
for x in range(Z.shape[1]):
    xcoordZ[x,:] = np.array([ i*ds-dis for i in range(dimpath) ])

# Define plot settings (parameter-specific) (for secondary variables)
plt_min_2 = -3
plt_max_2 = 3
plt_d_2   = 0.5
levels_2  = np.arange(plt_min_2, plt_max_2, plt_d_2)

#------------------------------------------------------------------------FIRST PLOT: CROSSSECTION

# Create figure (vertical cross section)
fig = plt.figure()
ax  = fig.add_subplot(1,1,1)

# Plot primary variable data
#ctf = ax.contourf(xcoord, vcross_p,
#                  vcross,
#                  levels = levels,
#                  cmap = cmap,
#                  norm=norm,
#                  extend = 'both')

# Plot secondary variable data
#ct = ax.contour(xcoord, vcross_p,
#                vcross_2,
#                levels = levels_2,
#                colors = 'grey',
#                linewidths = 1.5)
from matplotlib.colors import ListedColormap, BoundaryNorm

cmap = matplotlib.cm.nipy_spectral
levels = np.arange(0,60,5)
norm = BoundaryNorm(levels,256)
pv_levels=levels
h = ax.contourf(xcoord, vcross_p,
                vcross_U,
                levels = levels,
                cmap = cmap,
                norm=norm,
                extend = 'both')

#cons = ax.contour(xcoord,vcross_p,vcross_U,levels=np.array([5,10,20,30,40,50,60]),linewidths=1,colors='k')

#plt.clabel(cons, inline=1, fontsize=6, fmt='%d')
# Add contour labels
#ax.clabel(ct,
#          inline = True,
#          inline_spacing = 1,
#          fontsize = 10.,
#          fmt = '%.0f')


# Design axes
ax.set_ylabel('Pressure [hPa]', fontsize=12)
ax.set_ylim(bottom=ymin, top=ymax)
#ax.set_xlim(-500,500)
#ax.set_xticks(ticks=np.arange(-500,500,250))
# Invert y-axis
plt.gca().invert_yaxis()

# Add colorbar
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(h, ticks=pv_levels,cax=cbax)
func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
fig.canvas.mpl_connect('draw_event', func)
cbar.ax.set_xlabel(unit)

## Save figure
figname = 'U-of-' + pfile + '-%d.png'%lon_end
#print(figname)
fig.savefig(outpath+str(figname), bbox_inches = 'tight',dpi=300)
#plt.show()
#Close figure
plt.close(fig)


#fig = plt.figure()
#ax  = fig.add_subplot(1,1,1)
#cmap,pv_levels,norm,ticklabels=PV_cmap2()
#levels=pv_levels
#
#h = ax.contourf(xcoord, vcross_p,
#                vcross_PV,
#                levels = levels,
#                cmap = cmap,
#                norm=norm,
#                extend = 'both')
#
#cons = ax.contour(xcoord,vcross_p,vcross_TH,levels=np.arange(290,420.1,5),linewidths=0.5,colors='k')
#
#plt.clabel(cons, inline=1, fontsize=6, fmt='%d')
#
#ax.set_ylabel('Pressure [hPa]', fontsize=12)
#ax.set_ylim(bottom=ymin, top=ymax)
##ax.set_xlim(-500,500)
##ax.set_xticks(ticks=np.arange(-500,500,250))
## Invert y-axis
#plt.gca().invert_yaxis()
#
## Add colorbar
#cbax = fig.add_axes([0, 0, 0.1, 0.1])
#cbar=plt.colorbar(h, ticks=pv_levels,cax=cbax)
#func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
#fig.canvas.mpl_connect('draw_event', func)
#cbar.ax.set_xlabel(unit)
#
### Save figure
#figname = 'PV-of-' + pfile + '-%d.png'%lon_end
##print(figname)
#fig.savefig(outpath+str(figname), bbox_inches = 'tight',dpi=300)
##plt.show()
##Close figure
#plt.close(fig)



fig = plt.figure()
ax  = fig.add_subplot(1,1,1)

# Plot primary variable data
#ctf = ax.contourf(xcoord, vcross_p,
#                  vcross,
#                  levels = levels,
#                  cmap = cmap,
#                  norm=norm,
#                  extend = 'both')

# Plot secondary variable data
#ct = ax.contour(xcoord, vcross_p,
#                vcross_2,
#                levels = levels_2,
#                colors = 'grey',
#                linewidths = 1.5)
from matplotlib.colors import ListedColormap, BoundaryNorm

cmap = matplotlib.cm.nipy_spectral
levels = np.arange(1e3,1.5001e5,2e3)
norm = BoundaryNorm(levels,256)
pv_levels=levels
h = ax.contourf(xcoordZ, vcross_pZ,
                vcross_Z,
                levels = levels,
                cmap = cmap,
                norm=norm,
                extend = 'both')

cons = ax.contour(xcoord,vcross_p,vcross_U,levels=np.array([5,10,20,30,40,50,60]),linewidths=1,colors='k')

plt.clabel(cons, inline=1, fontsize=6, fmt='%d')

# Design axes
ax.set_ylabel('Pressure [hPa]', fontsize=12)
ax.set_ylim(bottom=ymin, top=ymax)
plt.gca().invert_yaxis()

# Add colorbar
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(h, ticks=pv_levels,cax=cbax)
func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
fig.canvas.mpl_connect('draw_event', func)
cbar.ax.set_xlabel(unit)

figname = 'GHT-of-' + pfile + '-%d.png'%lon_end
fig.savefig(outpath+str(figname), bbox_inches = 'tight',dpi=300)
plt.close(fig)
