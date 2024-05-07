import numpy as np
import argparse
from netCDF4 import Dataset as ds
from wrf import getvar,interplevel

parser = argparse.ArgumentParser(description="perturb unperturbed reference domain and write new output field")
parser.add_argument('sim',default='',type=str,help='which reference state to perturb: mean or overlap')
parser.add_argument('time',default='',type=str,help='choose time stamp')
parser.add_argument('lon1',default='',type=float,help='choose time stamp')
parser.add_argument('lon2',default='',type=float,help='choose time stamp')
parser.add_argument('lat1',default='',type=float,help='choose time stamp')
parser.add_argument('lat2',default='',type=float,help='choose time stamp')

args = parser.parse_args()

sim=str(args.sim)
time=str(args.time)
lon1 = float(args.lon1)
lon2 = float(args.lon2)
lat1 = float(args.lat1)
lat2 = float(args.lat2)

b='/atmosdyn2/ascherrmann/013-WRF-sim/' + sim + '/'

p = b + 'wrfout_d01_2000-12-%s:00:00'%time

d=ds(p,mode='r')
pres = getvar(d,'pressure',meta=False)
zh = getvar(d,'z',meta=False)
lon2d = getvar(d,'lon',meta=False)
lat2d = getvar(d,'lat',meta=False)

lon3d = np.tile(lon2d,(pres[:,0,0].size,1,1))
lat3d = np.tile(lat2d,(pres[:,0,0].size,1,1))

z,y,x = np.where((lon3d>=lon1) & (lon3d<=lon2) & (lat3d>=lat1) & (lat3d<=lat2) & (pres<=950) & (pres>=600))

zeva = zh[z,y,x]

dxl,dyl=0.5,0.5

x=lon2d[0,x]
y=lat2d[y,0]

save=np.zeros((x.size,3))
save[:,0]=x
save[:,1]=y
save[:,2]=zeva

np.savetxt(b+'wcb_start.ll',save,fmt='%.2f',delimiter=' ',newline='\n')


