from netCDF4 import Dataset as ds
import numpy as np
import os

seasons = ['DJF','MAM','JJA','SON']
names = ['west','east','south','north']
km=['200','400','800']
ofsetfac=[0.5,1,2]
xof=[-8,8,0,0]
yof=[0,0,-8,8]

#roi
ics='/atmosdyn2/ascherrmann/013-WRF-sim/ics/'

for sea in seasons:
    d = ds(ics + '%s-clim/met_em.d01.2000-12-01_00:00:00.nc'%(sea),'r')

    U=np.squeeze(d.variables['UU'][:])
    ght=np.squeeze(d.variables['GHT'][:])

    zmax,ymax,xmax = np.where(U[:26,:,:170]==np.max(U[:26,:,:170]))

    print(zmax,ymax,xmax)
    os.system('module load dyn_tools;module load conda/2022;source activate iacpy3_2022; bash /atmosdyn2/ascherrmann/scripts/WRF/pvinv-cart/ERA5-paper-check-jet-automatic.sh %s %d %d %d %s %s '%(sea,int(xmax+1),int(ymax+1),int(ght[zmax,ymax,xmax]),'max',0))
    for k,off in zip(km,ofsetfac[:]):
        for n,xo,yo in zip(names,xof,yof[:]):
            # fortran starts with index 1
            xtmp = xmax + int(xo*off) + 1
            ytmp = ymax + int(yo*off) + 1

            ztmp=ght[zmax,ytmp,xtmp]

            print('%s %d %d %d %s %s '%(sea,int(xtmp),int(ytmp),int(ztmp),n,k))
            os.system('module load dyn_tools;module load conda/2022;source activate iacpy3_2022; bash /atmosdyn2/ascherrmann/scripts/WRF/pvinv-cart/ERA5-paper-check-jet-automatic.sh %s %d %d %d %s %s'%(sea,int(xtmp),int(ytmp),int(ztmp),n,k))

os.system('module load dyn_tools; rsync -artv /atmosdyn2/ascherrmann/scripts/WRF/pvinv-cart/perturbed_files ascherrmann@euler:/cluster/work/climate/ascherrmann/ics/')

    

