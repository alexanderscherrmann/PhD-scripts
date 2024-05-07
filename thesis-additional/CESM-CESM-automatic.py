from netCDF4 import Dataset as ds
import numpy as np
import os

seasons = ['DJF','MAM','SON']
period=['2010','2040','2070','2100']
names = ['west']#,'east','south','north']
km=['200']#,'400','800']
ofsetfac=[0.5,1,2]
xof=[-8,8,0,0]
yof=[0,0,-8,8]

#roi
x0,y0,x1,y1=70,30,181,101

ics='/atmosdyn2/ascherrmann/013-WRF-sim/ics/'

for sea in seasons[1:2]:
  for perio in period[:1]:
    d = ds(ics + 'CESM-%s-%s-clim/met_em.d01.2000-12-01_00:00:00.nc'%(perio,sea),'r')

    u=np.squeeze(d.variables['UU'][:])
    p=np.squeeze(d.variables['PRES'][:])
    pl=np.where(p[:,0,0]==300*100)[0][0]
    ght=np.squeeze(d.variables['GHT'][:])

    u300roi=u[pl,y0:y1,x0:x1]
    ght300=ght[pl]
    ymax,xmax=np.where(u300roi==np.max(u300roi))[0][0],np.where(u300roi==np.max(u300roi))[1][0]

    xmax+=x0
    ymax+=y0

#    os.system('module load dyn_tools;module load conda/2022;source activate iacpy3_2022; bash /atmosdyn2/ascherrmann/scripts/WRF/pvinv-cart/CESM-CESM-automatic.sh %s %d %d %d %s %s %s'%(sea,int(xmax+1),int(ymax+1),int(ght300[ymax,xmax]),'max',0,perio))
    for k,off in zip(km,ofsetfac[:]):
        for n,xo,yo in zip(names,xof,yof[:]):
            # fortran starts with index 1
            xtmp = xmax + int(xo*off) + 1
            ytmp = ymax + int(yo*off) + 1

            ztmp=ght300[ytmp,xtmp]

            print('%s %d %d %d %s %s '%(sea,int(xtmp),int(ytmp),int(ztmp),n,k))
            os.system('module load dyn_tools;module load conda/2022;source activate iacpy3_2022; bash /atmosdyn2/ascherrmann/scripts/WRF/pvinv-cart/CESM-CESM-automatic.sh %s %d %d %d %s %s %s'%(sea,int(xtmp),int(ytmp),int(ztmp),n,k,perio))

os.system('module load dyn_tools; rsync -artv /atmosdyn2/ascherrmann/scripts/WRF/pvinv-cart/perturbed_files-CESM ascherrmann@euler:/cluster/work/climate/ascherrmann/ics/')

    

