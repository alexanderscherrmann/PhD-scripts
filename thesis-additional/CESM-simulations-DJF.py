from netCDF4 import Dataset as ds
import numpy as np
import os

seasons = ['DJF','MAM','SON']
names = ['west','east','south','north']
km=['800','1200','1600']
ofsetfac=[2,3,4]
xof=[-8,8,0,0]
yof=[0,0,-8,8]

#roi
x0,y0,x1,y1=70,30,181,101

ics='/atmosdyn2/ascherrmann/013-WRF-sim/ics/'

for sea in seasons[:1]:

    d = ds(ics + '%s-clim/met_em.d01.2000-12-01_00:00:00.nc'%sea,'r')

    u=np.squeeze(d.variables['UU'][:])
    p=np.squeeze(d.variables['PRES'][:])
    pl=np.where(p[:,0,0]==300*100)[0][0]
    ght=np.squeeze(d.variables['GHT'][:])

    u300roi=u[pl,y0:y1,x0:x1]
    ght300=ght[pl]
    ymax,xmax=np.where(u300roi==np.max(u300roi))

    xmax+=x0
    ymax+=y0
    for k,off in zip(km,ofsetfac[:]):
        for n,xo,yo in zip(names,xof,yof[:]):
            # fortran starts with index 1
            xtmp = xmax + xo*off + 1
            ytmp = ymax + yo*off + 1

            ztmp=ght300[ytmp,xtmp]

#            print('%s %d %d %d %s %s '%(sea,int(xtmp),int(ytmp),int(ztmp),n,k))
            os.system('bash /atmosdyn2/ascherrmann/scripts/WRF/pvinv-cart/large-horizontal-shifts.sh %s %d %d %d %s %s '%(sea,int(xtmp),int(ytmp),int(ztmp),n,k))


    

