import numpy as np
import pickle
import netCDF4
import os
from shutil import copyfile
import argparse
from wrf import interplevel as intp

def calc_RH(Q,T,p):
        return 0.263 * p * Q/ (np.exp(17.67 * (T-273.16)/(T-29.65)))

def readcdf(ncfile,varnam):
    infile = netCDF4.Dataset(ncfile, mode='r')
    var = infile.variables[varnam][:]
    return(var)

pl = '/atmosdyn2/ascherrmann/013-WRF-sim/'
pi = pl + 'ics/'
pd = '/home/ascherrmann/scripts/WRF/data/'

pres = np.array(['1', '2', '3',
            '5', '7', '10',
            '20', '30', '50',
            '70', '100', '125',
            '150', '175', '200',
            '225', '250', '300',
            '350', '400', '450',
            '500', '550', '600',
            '650', '700', '750',
            '775', '800', '825',
            '850', '875', '900',
            '925', '950', '975',
            '1000'])
pres = pres.astype(int)

metf = netCDF4.Dataset(pi +fol + '/' + 'met_em.d01.2000-12-01_00:00:00.nc',mode='r')
varnc = list(metf.variables.keys())[1:]

fac=1.3

seasons = ['DJF','MAM','JJA','SON']
for sea in seasons:
    if not os.path.isdir(pi + sea + '-adjclim'):
        os.mkdir(pi + sea + '-adjclim')

    pa = sea + '-adjclim/'
    cpp = pi + sea + '-clim/'
    for d in os.listdir(cpp):
        copyfile(cpp+d,pi+pa+d)
    
    pc = pi + pa
    nctmp =dict()
    
    
    nc = netCDF4.Dataset(pc + 'met_em.d01.2000-12-01_00:00:00.nc',mode='r+')
    
    U3dc = nc['UU'][0] * fac
    p3d = np.zeros_like(nc['TT'][0])
    p3d[0] = nc['PSFC']
    for w,p in np.flip(pres):
        for j in range(p3d.shape[1]-1):
            for i in range(p3d.shape[2]-1):
                const = 0.5 * f[j]/9.81  * 55588.74 
                p3d[w+1,j+1,i+1] = p*100 - const *  nc['UU'][0,w,j+1,i+1] * (p*100-p3d[w,j+1,i+1])/(nc['GHT'][w+1,j+1,i+1]-nc['GHT'][w,j+1,i+1])
    rho = -1 * (p3d[1:]-p3d[:-1])/9.81
    T3d = p3d[1:]/rho/287.058

    for var in ['UU','VV','TT','RH','GHT']:
    for w,p in np.flip(pres):
        nc['VV'][0,1:,:-1] = intp(nc['VV'][0,1:,:-1],p3d,p,meta=False)
        nc['UU'][0,1:,:,:-1] = intp(nc['UU'][0,1:,:,:-1],p3d,p,meta=False)
        nc['TT'][0,
    

                

    



