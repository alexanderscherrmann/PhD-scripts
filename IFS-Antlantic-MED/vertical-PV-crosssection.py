import sys,argparse
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
import helper
from useful_functions import create_lonlat_from_file,resize_colorbar_horz,resize_colorbar_vert
from colormaps import PV_cmap2
from conversions import coord2grid
from conversions import level_to_index_T
import netCDF4 as nc
import load_netcdf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os.path
import xarray as xr
#dypy
from dypy.small_tools import CrossSection
from dypy.netcdf import read_var_bbox
from dypy.small_tools import interpolate
from dypy.tools.py import print_args,ipython


import pickle
import netCDF4
import numpy as np
import math
import dypy.netcdf as nc


def lonlat_from_P(filename, name_fld, linear=None):
    ncfile = netCDF4.Dataset(filename)
    ymin=np.int(np.squeeze(ncfile.variables[name_fld].ymin))
    ymax=np.int(np.squeeze(ncfile.variables[name_fld].ymax))
    xmin=np.int(np.squeeze(ncfile.variables[name_fld].xmin))
    xmax=np.int(np.squeeze(ncfile.variables[name_fld].xmax))
    zmin=np.int(np.squeeze(ncfile.variables[name_fld].zmin))
    zmax=np.int(np.squeeze(ncfile.variables[name_fld].zmax))

    dim=nc.read_dimensions(filename)
    xlen=[dim[i] for i in dim.keys()][0]
    ylen=[dim[i] for i in dim.keys()][1]
    zlen=[dim[i] for i in dim.keys()][4]

    z=np.linspace(np.min([zmin,zmax]),np.max([zmin,zmax]),zlen,dtype=int)
    lats = np.linspace(ymin,ymax,ylen)
    lons = np.linspace(xmin,xmax,xlen)

    if linear is None:
           lon,lat=np.meshgrid(lons,lats)
    else:
           lon=lons
           lat=lats

    ncfile.close()

    return lon,lat,z


#parser = argparse.ArgumentParser(description="horizontal slice of PV at pressure:")
#parser.add_argument('CYID', default=1, type=int, help='cyclone ID')

#args = parser.parse_args()

vari = ['PV', 'P', 'TH','THE','Q','T']
filetypes = ['S', 'S', 'S', 'S','P','P']

MONTHS = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
MONTHSN = np.arange(1,13,1)

p = '/home/ascherrmann/010-IFS/'
path=p
pload = '/home/ascherrmann/010-IFS/traj/MED/use/'
f = open(pload + 'PV-data-MEDdPSP-100-ZB-800PVedge-0.3.txt',"rb")
data = pickle.load(f)
datadi = data['rawdata']
f.close()
pload = '/home/ascherrmann/010-IFS/data/'
f = open(pload + 'All-CYC-entire-year-NEW-correct.txt',"rb")
data = pickle.load(f)
f.close()


#DATE = '20180303_09-017'
#DATE = '20171214_02-073'
DATE = '20180619_03-111'

LON=np.linspace(-180,180,901)
LAT=np.linspace(0,90,226)

pv_cmap,pv_levels,pvnorm,pvticklabels=PV_cmap2()
varS = np.array(['PV','THE'])
varP = np.array(['Q'])

for var in np.append(varS,'RH'):
    if(os.path.isdir(p + var + '/')==0):
        os.mkdir(p + var + '/')
    elif (os.path.isdir(p+var+'/'+'vertical/')==0):
        os.mkdir(p+var+'/'+'vertical/')
    
Var = dict()

labels = np.array(['PV',r'$\theta_e$','RH'])
units = np.array(['[PVU]','[K]',r'[%]'])
levels = np.array([pv_levels,np.arange(280,346,2),np.arange(0,105,5)])
ticklabels = np.array([pvticklabels,np.arange(280,346,6),np.arange(0,105,5)])
cmaps = np.array([pv_cmap,matplotlib.cm.bwr,matplotlib.cm.YlGnBu])
norms = np.array([pvnorm,plt.Normalize(np.min(levels[1]),np.max(levels[1])),plt.Normalize(np.min(levels[2]),np.max(levels[2]))])

for date in datadi.keys():
   if date==DATE:
    yyyy = int(date[0:4])
    MM = int(date[4:6])
    DD = int(date[6:8])
    hh = int(date[9:11])

    monthn = int(date[4:6])
    monthid, = np.where(MONTHSN==monthn)
    month = MONTHS[monthid[0]] + date[2:4]
    if month=='DEC18':
        month='NOV18'
    if DD<7 and int(date[-3:])>100:
        month = MONTHS[monthid[0]-1] + date[2:4]

    ana_path='/home/ascherrmann/010-IFS/data/'+ month + '/'

    clat = np.mean(data[month][int(date[-3:])]['clat'][np.where(data[month][int(date[-3:])]['dates']==date[:11])[0]]).astype(int)
    clon = np.mean(data[month][int(date[-3:])]['clon'][np.where(data[month][int(date[-3:])]['dates']==date[:11])[0]]).astype(int)

    ran=int(np.min(np.where(np.linspace(0,2*np.pi*6370,901)>1600)))
    latu = np.arange(clat-ran,clat+ran+1,1)
    lonu = np.arange(clon-ran,clon+ran+1,1)
    minlatc = np.min(latu)
    minlonc = np.min(lonu)

    pltlat =np.linspace(0,90,226)[latu]
    pltlon = np.linspace(-180,180,901)[lonu]
    minlatc = np.min(latu)
    maxlatc = np.max(latu)
    minpltlatc = pltlat[0]
    maxpltlatc = pltlat[-1]
    minlonc = np.min(lonu)
    maxlonc = np.max(lonu)
    minpltlonc = pltlon[0]
    maxpltlonc = pltlon[-1]

    pltlatcenter = np.linspace(0,90,226)[clat]
    pltloncenter = np.linspace(-180,180,901)[clon]
    vari=list(vari)
    filetypes=list(filetypes)
    
    lon1=round(minpltlonc,2)
    lat1=round(pltlatcenter,2)
    lon2=round(maxpltlonc,2)
    lat2=round(pltlatcenter,2)
    
    kl = np.linspace(minpltlonc,maxpltlonc,1001)
    centerid, = np.where(abs(kl[:]-pltloncenter)==np.min(abs(kl[:]-pltloncenter)))
    
    coos=((lon1,lat1),(lon2,lat2))

    f = ana_path+'S'+date[:11]
    filename=f

    lon,lat,z=lonlat_from_P(f,'P')
    lon1d,lat1d,z=lonlat_from_P(f,'P',linear=1)
    
    resolution=np.diff(lon)[0,0]
    
    lonmin=min([lon1,lon2])-resolution*3
    lonmax=max([lon1,lon2])+resolution*3
    latmin=min([lat1,lat2])-resolution*3
    latmax=max([lat1,lat2])+resolution*3
    
    pressure=np.arange(100.,1001.,25)
        #remove duplicate filetypes
    filetypes_all=list(set(filetypes))
        ### connect filetypes and variables in dict ###
    vardict=dict()
    for i,ftype in enumerate(filetypes):
                    #create key if not there yet
        if ftype not in vardict:
           vardict[ftype]={}
            #append variables to dictionary
        vardict[ftype][vari[i]]=np.array([])
        varlist_str=vardict[ftype].keys()
        varlist_str=sorted(varlist_str)

        filename = ana_path+ftype+date[:11]    

        if len(varlist_str)== 1:
                  blon,blat,var1,index= read_var_bbox(filename,varlist_str,(lonmin,lonmax,latmin,latmax), lon=lon,lat=lat,return_index=True)
                  varlist_plot=[var1]
        elif len(varlist_str)==2:
                  blon,blat,var1,var2,index = read_var_bbox(filename,varlist_str,(lonmin,lonmax,latmin,latmax),lon=lon,lat=lat, return_index=True)
                  varlist_plot=[var1,var2]
        elif len(varlist_str)==3:
                  blon,blat,var1,var2,var3,index = read_var_bbox(filename,varlist_str,(lonmin,lonmax,latmin,latmax),lon=lon,lat=lat, return_index=True)
                  varlist_plot=[var1,var2,var3]
        elif len(varlist_str)==4:
                  blon,blat,var1,var2,var3,var4,index = read_var_bbox(filename,varlist_str,(lonmin,lonmax,latmin,latmax), lon=lon,lat=lat,return_index=True)
                  varlist_plot=[var1,var2,var3,var4]
        elif len(varlist_str)==5:
                  blon,blat,var1,var2,var3,var4,var5,index = read_var_bbox(filename,varlist_str,(lonmin,lonmax,latmin,latmax),lon=lon,lat=lat, return_index=True)
                  varlist_plot=[var1,var2,var3,var4,var5]
        elif len(varlist_str)==6:
                  blon,blat,var1,var2,var3,var4,var5,var6, index = read_var_bbox(filename,varlist_str,(lonmin,lonmax,latmin,latmax),lon=lon,lat=lat, return_index=True)
                  varlist_plot=[var1,var2,var3,var4,var5,var6]
    
        i=0
        for var in varlist_str:
                       #deal with nans
               varlist_plot[i][varlist_plot[i]<=-999]=np.nan
                       #put to dictionary
               vardict[ftype][var]=varlist_plot[i]
               #raise running variable
               i+=1
    variables={}
        #add variables
    for ftype in vardict.keys():
       for var in vardict[ftype].keys():
                    #deal with pressure variable name
            if var=='P':
               name='p'
               #reverse first dimension of pressure values such that they are increasing along that dimension (required for CrossSection(...))
               variables[name]=vardict[ftype][var][::-1,:,:]
            else:
               name=var
               variables[var]=vardict[ftype][var][::-1,:,:]
        #add lat lon
    variables['lon']=lon1d[index[-1]]
    variables['lat']=lat1d[index[-2]]
    
    
    cross = CrossSection(variables,coos,pressure,version='regular',int2p=True)
    x, zi = np.meshgrid(cross.distances, cross.pressure)
    
    x = x - x[:,centerid]
    cross.Q = cross.Q/helper.qs(cross.T,zi * 100) * 100
    for q,variabl in enumerate(np.append(varS,varP)):
        tmp = getattr(cross,variabl)
        fig, ax = plt.subplots()
        ax.set_ylim(ymin=100, ymax=1000)
        ax.invert_yaxis()
        
        cf = ax.contourf(x, zi, tmp, levels=levels[q],cmap=cmaps[q],norm=norms[q], extend='both',zorder=0)
        
        th_spec=np.linspace(250,420, num=35)
        cs2=plt.contour(x,zi,cross.TH,th_spec,colors='#606060',linewidths=1)
        plt.clabel(cs2, inline=1, fontsize=8, fmt='%1.0f')
        if q>0:
            pvlevels=np.array([2, 2 + 1e-10])
            cs2 = plt.contour(x, zi, cross.PV, levels=pvlevels,colors='black',linewidths=1)
        ax.set_ylabel("pressure [hPa]")
        ax.set_xlabel("West-East distance to cyclone center [km]")
        
        cbax=fig.add_axes([0,0,0.1,0.1])
        cbar=plt.colorbar(cf, ticks=ticklabels[q], orientation='vertical',label=labels[q] + ' '  + units[q],cax=cbax)
        func=resize_colorbar_vert(cbax,ax)
        
        cbar.ax.set_xticklabels(ticklabels[q])
        func=resize_colorbar_vert(cbax,ax)
        fig.canvas.mpl_connect('draw_event',func)
        cbar.ax.set_yticklabels(ticklabels[q],fontsize=10)
        plt.draw()
        
        xticks=np.arange(-2400, 2401, 300)#np.round(np.min(x)),np.round(np.max(x)+1),np.round((np.max(x)-np.min(x))/5))/100.))
        xticks=list(xticks)
        if 0 not in xticks:
           xticks.append(0)
        xticks=sorted(xticks)
        while((xticks[0]<np.min(x)) or (xticks[-1]>np.max(x))):
            if (xticks[0]<np.min(x)):
                xticks = np.delete(xticks, 0)
            if (xticks[-1]>np.max(x)):
                xticks = np.delete(xticks, -1)
        ax.set_xticks(xticks)
    
        if variabl=='Q':
            variabl='RH'
    
        figname=path + variabl + '/' + 'vertical/' + variabl +  '-vertical-crosssection-'+date+'.png'
        
        fig.savefig(figname,dpi=300,bbox_inches="tight")
        plt.close()
    
