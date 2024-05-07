import os
import numpy as np

sets = ['genesis']#,'mature']
for se in sets:
    if se=='genesis':
        ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/4regionsPV/'
#        ra0,ra1 = 38,42
        ra0,ra1 = 6,10
    if se=='mature':
        ra0,ra1 = 46,50
        ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/'
    
    for dirs in os.listdir(ps):
        if dirs[-1]=='c' or dirs[-1]=='t':
            continue
        dates = np.array([])
        for f in os.listdir(ps + dirs + '/300/'):
            if f.startswith('D'):
                dates = np.append(dates,f)

        date = dates[ra0:ra1+1]
        cmd = 'cdo enssum '
        for d in date:
            cmd += '%s '%d

        cmd += 'SOM-4d-sumPV'
        os.chdir(ps + dirs + '/300/')
        os.system(cmd)
        os.system('cdo -divc,%d SOM-4d-sumPV SOM-4d-avPV'%date.size)




        

