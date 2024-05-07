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
    if not f.startswith('wrfout_d01_2000-12'):
        continue
    data = netCDF4.Dataset(p + f,'r+')
    if np.any(np.array(list(data.variables))=='PV'):
        print('skip',f)
        data.close()
        continue

    else:
        PV = wrf.getvar(data,'pvo',meta=False)
    
        time = data.dimensions['Time'].name
        bt = data.dimensions['bottom_top'].name
        sn = data.dimensions['south_north'].name
        we = data.dimensions['west_east'].name
        
        data.createVariable('PV','f8',(time,bt,sn,we))
        data.variables['PV'][0,:] = PV
        data.close()

