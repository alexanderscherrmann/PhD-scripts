from wrf import getvar as get
import numpy as np
from netCDF4 import Dataset as ds
import os
import argparse


parser = argparse.ArgumentParser(description=' ')
parser.add_argument('sim',default=0,type=str,help='')

args = parser.parse_args()
sim=str(args.sim)

p = '/atmosdyn2/ascherrmann/013-WRF-sim/' + sim + '/'

a = 0
for f in os.listdir(p):
    if f.startswith('wrfout'):
        a+=1
        d = ds(p + f,mode='a')

        if np.any(np.array(list(d.variables.keys()))=='PV'):
            d["PV"][0] = get(d,'pvo')
        else:
            d.createVariable('PV',"f4",("Time","bottom_top","south_north","west_east"))
            d["PV"][:] = np.zeros(d['P'][:].shape)
            d["PV"][0] = get(d,'pvo')

        d['PV'].FieldType = 104
        d['PV'].MemoryOrder = 'XYZ'
        d['PV'].description = 'potential vorticity in PVU'
        d['PV'].units = 'PVU'
        d['PV'].stagger = ''
        d['PV'].coordinates = 'XLONG XLAT XTIME'

        if np.any(np.array(list(d.variables.keys()))=='geopot'):
            d['geopot'][0] = (d['PH'][0,1:] + d['PHB'][0,1:])/9.80665
        else:
            d.createVariable('geopot',"f4",("Time","bottom_top","south_north","west_east"))
            d['geopot'][:] = np.zeros(d['P'][:].shape)
            d['geopot'][0] = (d['PH'][0,1:] + d['PHB'][0,1:])/9.80665
        
        d['geopot'].FieldType = 104
        d['geopot'].MemoryOrder = 'XYZ'
        d['geopot'].description = 'geopotential height in m'
        d['geopot'].units = 'm'
        d['geopot'].stagger = ''
        d['geopot'].coordinates = 'XLONG XLAT XTIME'

        if np.any(np.array(list(d.variables.keys()))=='geopots'):
            d['geopots'][0] = (d['PH'][0,0] + d['PHB'][0,0])/9.80665
        else:
            d.createVariable('geopots',"f4",("Time","south_north","west_east"))
            d['geopots'][:] = np.zeros(d['XLAT'][:].shape)
            d['geopots'][0] = (d['PH'][0,0] + d['PHB'][0,0])/9.80665

        d['geopots'].FieldType = 104
        d['geopots'].MemoryOrder = 'XYZ'
        d['geopots'].description = 'geopotential height at the surface'
        d['geopots'].units = 'm'
        d['geopots'].stagger = ''
        d['geopots'].coordinates = 'XLONG XLAT'
    
        d.close()
