import numpy as np
import netCDF4
import argparse
import os
import wrf

parser = argparse.ArgumentParser(description="composite vertical cross section of XX ocean below XXX hPa")
parser.add_argument('sim',default='',type=str,help='folder/simulation for which to evaluate surface pressure and PV at 300 hPa')

args = parser.parse_args()
sim=str(args.sim)

p = '/atmosdyn2/ascherrmann/013-WRF-sim/' + sim + '/'

for f in os.listdir(p):
    if not f.startswith('wrfout_d0'):
        continue

    data = netCDF4.Dataset(p + f,'r+')
    if np.any(np.array(list(data.variables))=='MSLP'):
        print('skip',f)
        data.close()
        continue
    else:

        SLP = wrf.getvar(data,'slp',meta=False)
    
        time = data.dimensions['Time'].name
        sn = data.dimensions['south_north'].name
        we = data.dimensions['west_east'].name
        
        data.createVariable('MSLP','f8',(time,sn,we))
        data.variables['MSLP'][0,:] = SLP
        data.close()

