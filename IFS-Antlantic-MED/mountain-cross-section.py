import numpy as np
import os
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')

from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import helper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patch
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle
import xarray as xr

from useful_functions import create_lonlat_from_file,resize_colorbar_horz,resize_colorbar_vert
from colormaps import PV_cmap2
from conversions import coord2grid
from conversions import level_to_index_T
import netCDF4 as nc
import load_netcdf
from dypy.small_tools import CrossSection
from dypy.netcdf import read_var_bbox
from dypy.small_tools import interpolate
from dypy.tools.py import print_args,ipython
import netCDF4
import math
import dypy.netcdf as nc


def colbar(cmap,minval,maxval,nlevels):
    maplist = [cmap(i) for i in range(cmap.N)]
    newmap = ListedColormap(maplist)
    norm = BoundaryNorm(pvr_levels,cmap.N)
    return newmap, norm


def calc_vperp(U,V,lat1,lat2,lon1,lon2):

        dlat=lat2-lat1
        dlon=lon2-lon1
        if dlat==0:
            return V
        elif dlon==0:
            return U
        else:
            angle=np.arctan(dlon/dlat)

            #compute total velocity
            vtot=np.sqrt(U**2+V**2)
            #compute angle of velocity
            angle_v=np.arccos(U/vtot)
            #compute component perpendicular to cross section
            v_perp=vtot*np.sin(angle-angle_v)

            return v_perp


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

def cal_projection_distance(x0,y0,xa,ya,xb,yb):
    xscal = (xa-x0)*(xb-x0)
    yscal = (ya-y0)*(yb-y0)
    xabs = np.sqrt((xa-x0)**2 + (ya-y0)**2)

    return (xscal + yscal)/xabs


CT = 'MED'

pload = '/home/ascherrmann/010-IFS/traj/' + CT + '/use/'

f = open(pload + 'PV-data-'+CT+'dPSP-100-ZB-800.txt','rb')
data = pickle.load(f)
f.close()

pload = '/home/ascherrmann/010-IFS/data/DEC17/'
psave = '/home/ascherrmann/010-IFS/'
maxv = 3000
minv = 800
pvr_levels = np.arange(minv,maxv,200)

data2 = xr.open_dataset(pload + 'IFSORO')
ap = plt.cm.BrBG
cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))

alpha=1.
linewidth=.2
ticklabels=pvr_levels

minpltlatc = 30
minpltlonc = -10

maxpltlatc = 50
maxpltlonc = 15

lon = data2['lon']
lat = data2['lat']
ZB = data2['ZB'].values[0,0]

oro = data['oro']
datadi = data['rawdata']
highOROid = data['highORO']

rdis = 400
H = 48
a = 1

xa1 = 7.65
xa2 = 9.5

ya1 = 45.25
ya2 = 46.55

dxa = xa2-xa1
dya = ya2-ya1

xa21 = 7.2
xa22 = 7.55

ya21 = 45.25
ya22 = 46.55

dxa2 = xa22 - xa21
dya2 = ya22 - ya21

ma = dya/dxa
ma2 = dya2/dxa2

xan = xa1-0.35
yan = ya1 + ma*(-0.35)
yan2 = ya2 + ma * (-0.25) + 0.4

marker = ['x','d','s','o','*','^','v','p','.']
color = ['k','r','b','orange','purple','dodgerblue','magenta','grey','green']

vari = ['PV', 'P', 'TH','V','U']
filetypes = ['S', 'S', 'S', 'P','P']
calvar = ['PV','TH','Vperp']


for q,date in enumerate(highOROid[1:]):
    idp = np.where(datadi[date]['PV'][:,0]>=0.75)[0]
    tralon = datadi[date]['lon'][idp,:]
    tralat = datadi[date]['lat'][idp,:]
    trap = datadi[date]['P'][idp,:]

    PVoro = oro[date]['env']['APVTOT'][idp,:]
    deltaPVoro = np.zeros(datadi[date]['time'][idp,:].shape)
    deltaPVoro[:,1:] = PVoro[:,:-1]-PVoro[:,1:]
    pvr = deltaPVoro

    loc= np.where(np.sum(pvr,axis=0)/len(idp)>0.1)[0]
    
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
    for u in range(len(loc)):
        ax.plot([],[],marker=marker[u],ls='',color=color[u])

    legends = np.array([])
    for u in range(len(loc)):
        loc2 = np.where(datadi[date]['ZB'][idp,loc[u]]>800)[0]
        ax.scatter(tralon[loc2,loc[u]],tralat[loc2,loc[u]],marker=marker[u],color=color[u],s=2,zorder=100)
        legends = np.append(legends,str(-1*loc[u]) + ' h')

    ax.plot([xan,xa2],[yan,yan2 ])
    ax.plot([7.2,7.55],[44.8,44.2])
    ax.coastlines() 
    lc=ax.contourf(lon,lat,ZB,levels=pvr_levels,cmap=cmap,extend='both',norm=norm,zorder=1)
    lc.cmap.set_under('white')
    
    lonticks=np.arange(minpltlonc, maxpltlonc,5)
    latticks=np.arange(minpltlatc, maxpltlatc,5)
    ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
    ax.set_yticks(latticks, crs=ccrs.PlateCarree());
    ax.set_xticklabels(labels=lonticks,fontsize=8)
    ax.set_yticklabels(labels=latticks,fontsize=8)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(lc, ticks=pvr_levels,cax=cbax)
    ax.legend(legends,loc='upper left') 
    func=resize_colorbar_vert(cbax, ax, pad=0.01, size=0.02)
    fig.canvas.mpl_connect('draw_event', func)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel('m',fontsize=8)
    cbar.ax.set_xticklabels(ticklabels)
    
    figname = psave + 'traj-hour-positions-' + date  + '.png'
    fig.savefig(figname,dpi=300,bbox_inches="tight")
    plt.close()


    datadi2 = dict()

    crossdates = np.array([])
    cmap,pvlvl,norm,ticklabels=PV_cmap2()

    yyyy = int(date[0:4])
    MM = int(date[4:6])
    DD = int(date[6:8])
    hh = int(date[9:11])
    search = date[:11] + '-ID-%06d'%int(date[-3:])
    for d in os.listdir('/home/ascherrmann/010-IFS/traj/MED/'):
        if d.startswith('traend-' + search):
            month = d[-9:-4]    

    for k in range(0,loc[-1]+1):
        if(np.any(loc==k)):
            crossdates=np.append(crossdates,str(yyyy) + '%02d'%MM + '%02d'%DD + '_' + '%02d'%hh)

        hh-=1
        if(hh<0):
            hh+=24
            DD-=1
            if(DD<1):
              #  DD+=
                MM-=1
                if(MM<1):
                    MM+=12
                    yyyy-=1

    datapath = '/home/ascherrmann/010-IFS/data/' + month 
    lon1 = xan
    lat1 = yan
    lon2 = xa2
    lat2 = yan2

    pltloncenter = (lon2+lon1)/2
    pltlatcenter = (lat2+lat1)/2
    coos=((lon1,lat1),(lon2,lat2))


    for qr,dato in enumerate(crossdates):
        vari=list(vari)
        filetypes=list(filetypes)
        
        f = datapath+'/S'+dato
        filename=f
        
        kl = np.linspace(minpltlonc,maxpltlonc,1001)
        centerid, = np.where(abs(kl[:]-pltloncenter)==np.min(abs(kl[:]-pltloncenter)))
    
    
        lon,lat,z=lonlat_from_P(f,'P')
        lon1d,lat1d,z=lonlat_from_P(f,'P',linear=1)
    
        resolution=np.diff(lon)[0,0]
    
        lonmin=min([lon1,lon2])-resolution*3
        lonmax=max([lon1,lon2])+resolution*3
        latmin=min([lat1,lat2])-resolution*3
        latmax=max([lat1,lat2])+resolution*3
    
        pressure=np.arange(100.,1001.,25)
    
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
    
            filename = datapath+'/'+ftype+dato
    
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
    
        v_perp=calc_vperp(variables['U'],variables['V'],lat1,lat2,lon1,lon2)
                #add to variables
        variables['Vperp']=v_perp

    
        cross = CrossSection(variables,coos,pressure,version='regular',int2p=True)

        for rt in calvar:
            tmp = getattr(cross,rt)
            if(qr==0):
                datadi2[rt] = np.ones(tmp.shape)*tmp
            else:
                datadi2[rt] += np.ones(tmp.shape)*tmp
    x,zi = np.meshgrid(cross.distances, cross.pressure)

    pldi = dict()
    for rt in calvar:
        pldi[rt] = datadi2[rt]/len(crossdates)

    fig, ax = plt.subplots()
    ax.set_ylim(ymin=100, ymax=1000)
    ax.invert_yaxis()

    cf = ax.contourf(x, zi, pldi['PV'], levels=pvlvl,cmap=cmap,norm=norm,extend='both')
    th_spec=np.linspace(250,420, num=35)
    cs2=plt.contour(x,zi,pldi['TH'],th_spec,colors='#606060',linewidths=1)
    plt.clabel(cs2, inline=1, fontsize=8, fmt='%1.0f')
    v_spec=np.arange(-95,100,10)
    cs2=plt.contour(x,zi,cross.Vperp,v_spec,colors='w',linewidths=1)
    plt.clabel(cs2, inline=1, fontsize=8, fmt='%2.0f')

    for u in range(len(loc)):
        loc2 = np.where( (datadi[date]['ZB'][idp,loc[u]]>800) & (tralon[:,loc[u]]>xan) & (tralon[:,loc[u]]<xa2) & (tralat[:,loc[u]]>yan) & (tralat[:,loc[u]]<yan2))[0]
        for zs in range(len(loc2)):
            dis = helper.convert_lon_lat_dis_to_radial_dis(cal_projection_distance(xan,yan,xa2,yan2,tralon[loc2[zs],loc[u]],tralat[loc2[zs],loc[u]]))
            ax.scatter(dis,trap[loc2[zs],loc[u]],marker='x',color='k',s=2,zorder=100)


    ax.set_ylabel("pressure [hPa]")
    cbax=fig.add_axes([0,0,0.1,0.1])
    cbar=plt.colorbar(cf, ticks=ticklabels, orientation='vertical',label='PV [PVU]',cax=cbax)
    func=resize_colorbar_vert(cbax,ax)
    cbar.ax.set_xticklabels(ticklabels)
    fig.canvas.mpl_connect('draw_event',func)
    cbar.ax.set_yticklabels(ticklabels,fontsize=10)
    plt.draw()

    xticks=np.array(np.round(np.arange(np.min(x),np.max(x)+1,np.round((np.max(x)-np.min(x))/5))/100.)*100,dtype=np.int)
    xticks=list(xticks)
    if 0 not in xticks:
       xticks.append(0)
    if np.max(x) not in xticks:
       xticks.append(np.max(x))
    xticks=sorted(xticks)

    #remove second and second last tick for beauty reasons :-) if the distances to start and end are to small
    if xticks[1]<np.diff(xticks)[0]/2.:
       xticks.remove(xticks[1])
    if xticks[-2]>np.max(x)-np.diff(xticks)[0]/2.:
       xticks.remove(xticks[-2])
    ax.set_xticks(xticks)

    #create xticklabels from xticks
    for n,i in enumerate(xticks):
        if i==0:
           xticks[n]=str(np.round(lat1,2))+'°N /\n'+str(np.round(lon1,2))+'°E'
        elif i==np.max(x):
           xticks[n]=str(np.round(lat2,2))+'°N /\n'+str(np.round(lon2,2))+'°E'
        else:
           xticks[n]=str(i)
    ax.set_xticklabels(xticks)


    figname='/home/ascherrmann/010-IFS/averaged-vertical-crosssection-alps1.png'
    fig.savefig(figname,dpi=300,bbox_inches="tight")
    plt.close()























