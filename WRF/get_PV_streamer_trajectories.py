import numpy as np
import netCDF4
import argparse
import wrf

parser = argparse.ArgumentParser(description="perturb unperturbed reference domain and write new output field")
parser.add_argument('sim',default='',type=str,help='which reference state to perturb: mean or overlap')
parser.add_argument('day',default='',type=str,help='which reference state to perturb: mean or overlap')

parser.add_argument('lon0',default=0,type=float,help='')
parser.add_argument('lon1',default=0,type=float,help='')

parser.add_argument('lat0',default=0,type=float,help='')
parser.add_argument('lat1',default=0,type=float,help='')

parser.add_argument('p0',default=0,type=float,help='')
parser.add_argument('p1',default=0,type=float,help='')
args = parser.parse_args()

sim=str(args.sim)
day=str(args.day)
lon0=float(args.lon0)
lon1=float(args.lon1)
lat0=float(args.lat0)
lat1=float(args.lat1)
p0=float(args.p0)
p1=float(args.p1)



b='/atmosdyn2/ascherrmann/013-WRF-sim/' + sim + '/'
p = b + 'wrfout_d01_' + day[:4] + '-' + day[4:6] + '-' + day[6:8] + '_' + day[9:] + ':00:00'

d=netCDF4.Dataset(p,mode='r')

pv=wrf.getvar(d,'pvo',meta=False)
height=wrf.getvar(d,'height',meta=False)
pressure=wrf.getvar(d,'pressure',meta=False)

lon=wrf.getvar(d,'lon')
lat=wrf.getvar(d,'lat')

lon=np.tile(lon,(pv.shape[0],1,1))
lat=np.tile(lat,(pv.shape[0],1,1))

z,y,x = np.where((lon>=lon0) & (lon<=lon1) & (lat>=lat0) & (lat<=lat1) & (pressure<=p0) & (pressure>=p1) & (pv>=3.0))

z=np.array(z)
x=np.array(x)
y=np.array(y)

zh=height[z,y,x]
xx=lon[z,y,x]
yy=lat[z,y,x]

save=np.zeros((len(x),3))
save[:,0]=xx
save[:,1]=yy
save[:,2]=zh

np.savetxt(b+'startf.ll',save,fmt='%.1f',delimiter=' ',newline='\n')
