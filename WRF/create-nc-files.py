###
### creates PV field in netcdf file that should be inverted
###
import numpy as np
import netCDF4

def PV_anomaly_sphere(nx,ny,nz,r,field,ano):
    tmpf = np.zeros(field.shape)
    #center of anomaly here spherical
    xc,yc,zc = int((nx-1)/2),int((ny-1)/2),int((nz-1)/2)
    x = np.linspace(0, (nx-1) * resx,nx)
    x -= x[xc]
    y = np.linspace(0, (ny-1) * resy,ny)
    y -= y[yc]
    z = np.linspace(0, (nz-1) * resz,nz)
    z -= z[zc]

    xx,yy,zz = np.meshgrid(x,y,z)
    rr = np.sqrt(xx**2  + yy**2 + zz**2)
    xi,yi,zi = np.where(rr<=r)
    for q,w,e in zip(xi,yi,zi):
        tmpf[0,e,w,q] += ano[0,e,w,q]

    return tmpf

## path and filename
p = '/home/ascherrmann/scripts/WRF/pvinv-cart/'
nco = netCDF4.Dataset(p + 'PVtest',mode='w')

# predefined dimension name
dimname = 'QGPV'

#resolutions
nz = 51
ny = 101
nx = 101

resx = 55588.74
resy = 55588.74    #m
resz = 125.         #m

## create the file
nco.createDimension("time",1)
nco.createDimension("dimz_" + dimname,nz)
nco.createDimension("dimy_" + dimname,ny)
nco.createDimension("dimx_" + dimname,nx)

## attributes accessed by nco.ncattrs()
nco.domxmin = -1 *(nx-1)/2
nco.domxmax = -1 * nco.domxmin
nco.domymin = -1 *(ny-1)/2
nco.domymax = -1 * nco.domymin
nco.domzmin = 0
nco.domzmax = (nz-1) * resz
nco.domamin = 0 
nco.domamax = 0
nco.constants_file_name = 'no_constants_file'

## anomaly characteristics
## can be constant or a function of distance
PVano = 2e-6 *  np.ones((1,nz,ny,nx)) #2 PVU
rano = 200000 # 200km
PVfield = np.zeros((1,nz,ny,nx))
PVfield += PV_anomaly_sphere(nx,ny,nz,rano,PVfield,PVano)


## write variable 
nco.createVariable('QGPV',"f4",("time","dimz_" + dimname,"dimy_"+ dimname,"dimx_" + dimname))
nco['QGPV'][:] = PVfield

### save file
nco.close()

