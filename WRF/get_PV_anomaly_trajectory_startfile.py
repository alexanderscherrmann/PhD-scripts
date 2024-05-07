import numpy as np
import netCDF4
import argparse


parser = argparse.ArgumentParser(description="perturb unperturbed reference domain and write new output field")
parser.add_argument('sim',default='',type=str,help='which reference state to perturb: mean or overlap')
parser.add_argument('per',default='',type=str,help='choose perturbation')

args = parser.parse_args()

sim=str(args.sim)
pert=str(args.per)
b='/atmosdyn2/ascherrmann/013-WRF-sim/' + sim + '/'
p='/home/ascherrmann/scripts/WRF/pvinv-cart/perturbations/' + pert

d=netCDF4.Dataset(p,mode='r')
QGPV=d.variables['QGPV'][0,:]

dxl,dyl,dz=0.5,0.5,200
#dx,dy=55588.74,55588.74
z,y,x=np.where(QGPV>=0.9*np.max(QGPV))

x0,y0=-120,10

x=x0+x*dxl
y=y0+y*dyl
z=z*dz

save=np.zeros((len(x),3))
save[:,0]=x
save[:,1]=y
save[:,2]=z

np.savetxt(b+'startf.ll',save,fmt='%.1f',delimiter=' ',newline='\n')


