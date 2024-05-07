import numpy as np
import wrf
from netCDF4 import Dataset as ds

pb = '/atmosdyn2/era5/cdf/'

Lat = np.arange(-90,90.1,0.5)
Lon = np.arange(-180,180.1,0.5)

region = [-8, 25, 28, 50]

regions=[[15,37,32,50],[1,20,36,57],[3,20,37,58]]
dates = ['20040122_14','20191222_10','20191213_20']

#for date in ['10_00','11_00','11_12','12_00']:
for date,region in zip(dates,regions):
    loci,laci = np.where((Lon>=region[0]) & (Lon<=region[1]))[0],np.where((Lat>=region[2]) & (Lat<=region[3]))[0]
    p = pb + '%s/%s/'%(date[:4],date[4:6])

    sfile=p+'S%s'%date
    s = ds(sfile,'r')
    
    PS = s.variables['PS'][0,laci[0]:laci[-1]+1,loci[0]:loci[-1]+1]
    pv = s.variables['PV'][0,:,laci[0]:laci[-1]+1,loci[0]:loci[-1]+1]

    ak=s.variables['hyam'][137-98:]
    bk=s.variables['hybm'][137-98:]

    s.close()

    ps3d=np.tile(PS[:,:],(len(ak),1,1))
    p3d=(ak/100.+bk*ps3d.T).T
    
    pv300=wrf.interplevel(pv,p3d,300,meta=False)
    lai,loi=np.where(pv300>=2)
    
    plat=Lat[laci[lai]]
    plon=Lon[loci[loi]]
    
    pt=np.ones_like(lai)*300
    save = np.zeros((pt.size,4))
    save[:,1] = plon
    save[:,2] = plat
    save[:,3] = pt
    
    np.savetxt('/home/ascherrmann/fita-case-traj/fita-case-trajectories-%s.txt'%date,save,fmt='%.2f',delimiter=' ',newline='\n')
    
