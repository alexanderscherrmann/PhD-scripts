import numpy as np
import pickle
import netCDF4
import os
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser(description=' ')
parser.add_argument('fol',default=0,type=str,help='')

args = parser.parse_args()
fol=str(args.fol)


def calc_RH(Q,T,p):
	return 0.263 * p * Q/ (np.exp(17.67 * (T-273.16)/(T-29.65)))

def readcdf(ncfile,varnam):
    infile = netCDF4.Dataset(ncfile, mode='r')
    var = infile.variables[varnam][:]
    return(var)

LON = np.linspace(-180,179.5,720)
LAT = np.linspace(-90,90,361)

#fs = ['mean','-jet-overlap']
pres = np.array(['1', '2', '3',
            '5', '7', '10',
            '20', '30', '50',
            '70', '100', '125',
            '150', '175', '200',
            '225', '250', '300',
            '350', '400', '450',
            '500', '550', '600',
            '650', '700', '750',
            '775', '800', '825',
            '850', '875', '900',
            '925', '950', '975',
            '1000'])
pres = pres.astype(int)

pl = '/atmosdyn2/ascherrmann/013-WRF-sim/'
pi = pl + 'ics/'
pd = '/atmosdyn2/ascherrmann/scripts/WRF/data/'

#paths = ['ref-mean-ERA5-large/','ref-overlap-large/']
#paths=['ref-mean-ERA5-medium/','ref-overlap-medium/']
#paths = ['ref-mean/','ref-overlap/']

metf = netCDF4.Dataset(pi +fol + '/' + 'met_em.d01.2000-12-01_00:00:00.nc',mode='r')
varnc = list(metf.variables.keys())[1:]

# domain boundaries
doN, doW, doE, doS = np.max(metf.corner_lats),np.min(metf.corner_lons),np.max(metf.corner_lons),np.min(metf.corner_lats)
metf.close()

varname = ['var39','var40','var41','var42','var139','var170','var183','var236']
parameter = ['SM000007','SM007028','SM028100','SM100289','ST000007','ST007028','ST028100','ST100289']
###

## take U10M,V10M,D2M,T2M from ERA5 aswell as soil moisture

#varname2 = ['PS','MSL','SSTK','U10M','V10M','D2M','T2M']
varname2 = ['U10M','V10M','D2M','T2M']
###

# 2D cesm data
var2c = ['PS','PSL','SST','TS']
param2c = ['PSFC','PMSL','SST','SKINTEMP']

varp = ['Z3','RH','T','U','V']
paramP = ['GHT','RH','T','U','V']

seasons = ['DJF','MAM','JJA','SON']
period='2010'
for sea in seasons[:1]:
  if sea=='JJA':
      continue
  for memb in ['0900','1000','1100','1200','1300']: 

    if not os.path.isdir(pi + 'CESM-%s-2010-'%memb+ sea + '-clim'):
        os.mkdir(pi + 'CESM-%s-2010-'%memb + sea + '-clim')

    pa = 'CESM-%s-2010-'%memb + sea + '-clim/'
    
    ### copy original metgrid files into folder to run
    cpp = pi + fol + '/'
    for d in os.listdir(cpp):
        copyfile(cpp+d,pi+pa+d)
    
    pc = pi + pa
    
    Vars = dict()
    nctmp =dict()
    
    ###
    ### 2D fields downloaded take 'SM000007','SM007028','SM028100','SM100289','ST000007','ST007028','ST028100','ST100289' from ERA5
    ###
        
    for va in varname:
        Vars[va] = readcdf(pd + 'sfmean-ERA5-' + sea + '.nc',va)[0]
    
    lonsoil = readcdf(pd + 'sfmean-ERA5-' + sea + '.nc','lon')
    latsoil = readcdf(pd + 'sfmean-ERA5-' + sea + '.nc','lat')
    ###
    ### 2D fields from thermo era5 data 'U10M','V10M','D2M','T2M'
    ###
    
    for va in varname2:
        Vars[va] = readcdf(pd + 'Bmean-ERA5-' + sea,va)[0]
    
    lonN = readcdf(pd + 'Bmean-ERA5-' + sea,'lon')
    latN = readcdf(pd + 'Bmean-ERA5-' + sea,'lat')
    
    ###
    ### 3D fields U,GHT,V,T,RH
    ###
    
    p3dCESM = '/atmosdyn2/ascherrmann/015-CESM-WRF/%s/%s/2010/'%(sea,memb)

    for paraP, va in zip(paramP,varp):
        Vars[paraP] = readcdf(p3dCESM + 'period-mean-0.5-%s-%s.nc'%(period,sea),va)[0]
    
    lonP = readcdf(p3dCESM + 'period-mean-0.5-%s-%s.nc'%(period,sea),'lon')
    latP = readcdf(p3dCESM + 'period-mean-0.5-%s-%s.nc'%(period,sea),'lat')
    
    ###
    ### get right indeces
    ###
        
    losoil = np.where((lonsoil>=doW) & (lonsoil<=doE))[0]
    lasoil = np.where((latsoil<=doN) & (latsoil>=doS))[0]
    
    loP = np.where((lonP>=doW) & (lonP<=doE))[0]
    laP = np.where((latP<=doN) & (latP>=doS))[0]

    loN = np.where((lonN>=doW) & (lonN<=doE))[0]
    laN = np.where((latN<=doN) & (latN>=doS))[0]
    
    T = Vars['T']
    U = Vars['U']
    V = Vars['V']
    RH = Vars['RH']
    gp = Vars['GHT']

    
    loS0,loS1,laS0,laS1 = losoil[0],losoil[-1],lasoil[0],lasoil[-1]
    loP0,loP1,laP0,laP1 = loP[0],loP[-1],laP[0],laP[-1]
    loN0,loN1,laN0,laN1 = loN[0],loN[-1],laN[0],laN[-1]
    
    ### downloaded data starts from N to S hence reverse the index order
    nc = netCDF4.Dataset(pc + 'met_em.d01.2000-12-01_00:00:00.nc',mode='r+')
    ## data is saved in order from 1 hPa to 1000 hPa
    ## so index 0 corresponds to 1 hPa and -1 to 1000 hPa
    ## fill up netcdf from 1000 to 1 hPa thereforerevers order
    for w,p in enumerate(pres):
        nc['RH'][0,(w+1)] = RH[w,laP0:laP1,loP0:loP1]
        nc['UU'][0,w+1] = U[w,laP0:laP1,loP0:loP1+1]
        nc['VV'][0,(w+1)] = V[w,laP0:laP1+1,loP0:loP1]
        nc['TT'][0,(w+1)] = T[w,laP0:laP1,loP0:loP1]
        nc['GHT'][0,(w+1)] =gp[w,laP0:laP1,loP0:loP1]
    
    nc['TT'][0,0] = Vars['T2M'][laN0:laN1,loN0:loN1]
    nc['VV'][0,0] = Vars['V10M'][laN0:laN1+1,loN0:loN1]
    nc['UU'][0,0] = Vars['U10M'][laN0:laN1,loN0:loN1+1]
    nc['RH'][0,0] = 100 *( np.exp((17.625 * (Vars['D2M'][laN0:laN1,loN0:loN1]-273.16))/(243.04 + Vars['D2M'][laN0:laN1,loN0:loN1]-273.16))/np.exp((17.625*(nc['TT'][0,0]-273.16))/(243.04 + nc['TT'][0,0]-273.16)))
    
    ###
    ### 2D fields CESM PS,SST,MSL
    ###

    p2dCESM = '/atmosdyn2/ascherrmann/015-CESM-WRF/%s/%s/%s/'%(sea,memb,period)

    for para2D,va in zip(param2c,var2c):
        Vars[para2D] = readcdf(p2dCESM + 'period-surface-mean-0.5-%s-%s.nc'%(period,sea),va)[0]

    lon2D = readcdf(p2dCESM + 'period-surface-mean-0.5-%s-%s.nc'%(period,sea),'lon')
    lat2D = readcdf(p2dCESM + 'period-surface-mean-0.5-%s-%s.nc'%(period,sea),'lat')

    lo2D = np.where((lon2D>=doW) & (lon2D<=doE))[0]
    la2D = np.where((lat2D<=doN) & (lat2D>=doS))[0]

    lo20,lo21,la20,la21 = lo2D[0],lo2D[-1],la2D[0],la2D[-1]

    for pa in param2c:
        nc[pa][0]=Vars[pa][la20:la21,lo20:lo21]


    # where landmask is 0 but sst is 0 from CESM, take skin temperature as sst
    nc['SST'][0][(nc['LANDMASK'][0]==0) & (nc['SST'][0]==0)]=nc['SKINTEMP'][0][(nc['LANDMASK'][0]==0) & (nc['SST'][0]==0)]
    # before it was 0, maybe that makes a difference
    nc['SST'][0][nc['LANDMASK'][0]!=0]=np.nan


    ### soil fields have different shape than skintemperature
    for pa,va in zip(parameter,varname):
        nc[pa][0] = np.flip(Vars[va][0,laS0:laS1,loS0:loS1],axis=0)
    #### make every file the same
    for va in varnc:
        nctmp[va] = nc[va][:]
    nc.close() 
    
    ### copy perviously saved tmp stuff from initila file here
    for fl in os.listdir(pc):
        if fl.startswith('met_em.d01.2000'):
            nc = netCDF4.Dataset(pc + fl,mode='a')
        for va in varnc:
            nc[va][:] = nctmp[va]
        nc.close()

