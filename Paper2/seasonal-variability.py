from netCDF4 import Dataset as ds
import numpy as np
import os
import wrf

seasons = ['DJF','MAM','SON','JJA']

#roi
x0,y0,x1,y1=70,30,181,101

ics='/atmosdyn2/ascherrmann/013-WRF-sim/ics/'
dw='/atmosdyn2/ascherrmann/013-WRF-sim/'

for sea in seasons[:]:
    d = ds(ics + '%s-clim/met_em.d01.2000-12-01_00:00:00.nc'%(sea),'r')

    lon=np.squeeze(d.variables['CLONG'][0,0])+0.25
    lat=np.squeeze(d.variables['CLAT'][0,:,0])+0.25
    u=np.squeeze(d.variables['UU'][:])
    p=np.squeeze(d.variables['PRES'][:])

    dpv = ds(dw + '%s-clim/wrfout_d01_2000-12-01_00:00:00'%sea)
    pv=wrf.getvar(dpv,'pvo')
    pres = wrf.getvar(dpv,'pressure')
    pv300ic=wrf.interplevel(pv,pres,300,meta=False)

    pl=np.where(p[:,0,0]==300*100)[0][0]
    

    u300roi=u[pl,y0:y1,x0:x1]
    ymax,xmax=np.where(u300roi==np.max(u300roi))[0][0],np.where(u300roi==np.max(u300roi))[1][0]

    xmax+=x0
    ymax+=y0

    mlon=lon[xmax]
    mlat=lat[ymax]

    std=ds('/atmosdyn2/ascherrmann/paper/NA-MED-link/variability/climatology-wind-300-hPa-%s.nc'%sea.lower())
    p=ds('/atmosdyn2/era5/cdf/2001/10/P20011010_10')
    lon2=p.variables['lon']
    lat2=p.variables['lat']

    windvar=np.sqrt(np.squeeze(std.variables['VEL.VAR']))
    i,j=np.where(lon2==mlon)[0][0],np.where(lat2==mlat)[0][0]

    std=ds('/atmosdyn2/ascherrmann/paper/NA-MED-link/variability/std-PV-300-hPa-%s.nc'%sea.lower())
    pvvar=np.squeeze(std.variables['PV.VAR'])


    dp2 = ds(dw + '%s-clim-max-U-at-300-hPa-0.7-QGPV/wrfout_d01_2000-12-01_00:00:00'%sea)
    pv=wrf.getvar(dp2,'pvo')
    pres=wrf.getvar(dp2,'pressure')
    pv300per=wrf.interplevel(pv,pres,300,meta=False)

    deltaPV=pv300per-pv300ic
    
    print('10',sea,np.mean(windvar[j-10:j+11,i-10:i+11]),np.mean(pvvar[j-10:j+11,i-10:i+11]),np.mean(deltaPV[ymax-10:ymax+11,xmax-10:xmax+11]))
    print('5',sea,np.mean(windvar[j-5:j+6,i-5:i+6]),np.mean(pvvar[j-5:j+6,i-5:i+6]),np.mean(deltaPV[ymax-5:ymax+6,xmax-5:xmax+6]))
    print('15',sea,np.mean(windvar[j-15:j+16,i-15:i+16]),np.mean(pvvar[j-15:j+16,i-15:i+16]),np.mean(deltaPV[ymax-15:ymax+16,xmax-15:xmax+16]))
    print('15',sea,np.mean(windvar[j-20:j+21,i-20:i+21]),np.mean(pvvar[j-20:j+21,i-20:i+21]),np.mean(deltaPV[ymax-20:ymax+21,xmax-20:xmax+21]))


    

