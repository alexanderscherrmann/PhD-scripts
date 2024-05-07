import numpy as np
from wrf import interplevel as intp
from netCDF4 import Dataset as ds

import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import pickle
import os

import argparse

parser = argparse.ArgumentParser(description="composite vertical cross section of XX ocean below XXX hPa")
parser.add_argument('date',default='',type=str,help='folder/simulation for which to evaluate surface pressure and PV at 300 hPa')
parser.add_argument('lon0',default='',type=float,help='')
parser.add_argument('lat0',default='',type=float,help='')
parser.add_argument('lon1',default='',type=float,help='')
parser.add_argument('lat1',default='',type=float,help='')

args = parser.parse_args()
date=str(args.date)
lon0=float(args.lon0)
lat0=float(args.lat0)
lon1=float(args.lon1)
lat1=float(args.lat1)

p = '/atmosdyn2/ascherrmann/009-ERA-5/MED/'

Lat = np.arange(-90,90.1,0.5)
Lon = np.arange(-180,180.1,0.5)

#LON,LAT = np.meshgrid(Lon,Lat)

dates = [date]
#lon1,lat1,lon2,lat2 = -5,20,1.5,42
#lon3,lat3,lon4,lat4 = 1.5,20,50,48

lons = np.where((Lon>=lon0)&(Lon<=lon1))[0]
lats = np.where((Lat>=lat0)&(Lat<=lat1))[0]

#lons01 = np.where((Lon>1.5)&(Lon<=50))[0]
#lats01 = np.where((Lat>=20)&(Lat<=48))[0]
#lo00,lo01,la00,la01,lo10,lo11,la10,la11 = lons00[0],lons00[-1],lats00[0],lats00[-1],lons01[0],lons01[-1],lats01[0],lats01[-1]
#lons1,lats1 = np.where((LON>=-5) & (LAT>=20) & (LON<=1.5) & (LAT<=42))
#lons2,lats2 = np.where((LON>=1.5) & (LAT>=20) & (LON<=50) & (LAT<=48))

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

for k,t in enumerate(dates):
    yyyy = int(t[0:4])
    MM = int(t[4:6])
    DD = int(t[6:8])
    hh = int(t[9:])
    ana_path = '/atmosdyn2/era5/cdf/%d/%02d/'%(yyyy,MM)
    sfile = ana_path + 'S' + t

    s = ds(sfile)

    PS = s.variables['PS'][0,la0:la1,lo0:lo1]
    PV = s.variables['PV'][0,:,la0:la1,lo0:lo1]
    hyam = s.variables['hyam'][137-98:]
    hybm = s.variables['hybm'][137-98:]
    ps3d=np.tile(PS,(PV.shape[0],1,1))
    P=(hyam/100.+hybm*ps3d.T).T
    PV300 = intp(PV,P,300,meta=False)    

    x,y = np.where((PV300>=4))

    plon=Lon[lo0+x]
    plat=Lat[la0+y]
    pt=np.ones_like(plon)*300
    
    save = np.zeros((len(pt),4))
    save[:,1] = plon
    save[:,2] = plat
    save[:,3] = pt
    np.savetxt('/atmosdyn2/ascherrmann/013-WRF-sim/ERA5-streamer/trajectories-' + t + '.txt',save,fmt='%f', delimiter=' ', newline='\n')





