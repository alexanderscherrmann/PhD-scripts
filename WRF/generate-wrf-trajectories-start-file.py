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
pv = getvar(d,'pvo',meta=False)
z = getvar(d,'z',meta=False)
pres = getvar(d,'pressure',meta=False)
lon = getvar(d,'lon',meta=False)
lat = getvar(d,'lat',meta=False)

pv300=interplevel(pv,pres,300,meta=False)
z300=interplevel(z,pres,300,meta=False)

y,x = np.where((pv300>=2) & (lon>=lon1) & (lon<=lon2) & (lat>=lat1) & (lat<=lat2))

zeva = z300[y,x]

x0,y0=lon[0,0],lat[0,0]

dxl,dyl=0.5,0.5

x=lon[0,x]
y=lat[y,0]

save=np.zeros((len(x),3))
save[:,0]=x
save[:,1]=y
save[:,2]=zeva

np.savetxt(b+'startf.ll',save,fmt='%.2f',delimiter=' ',newline='\n')


