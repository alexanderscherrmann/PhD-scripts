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
pd = '/home/ascherrmann/scripts/WRF/test-overlap-ridge/'


metf = netCDF4.Dataset(pi +fol + '/' + 'met_em.d01.2000-12-01_00:00:00.nc',mode='r')
varnc = list(metf.variables.keys())[1:]

# domain boundaries
doN, doW, doE, doS = np.max(metf.corner_lats),np.min(metf.corner_lons),np.max(metf.corner_lons),np.min(metf.corner_lats)
metf.close()

varname = ['var235','var39','var40','var41','var42','var139','var170','var183','var236']
parameter = ['SKINTEMP','SM000007','SM007028','SM028100','SM100289','ST000007','ST007028','ST028100','ST100289']
###
varsfnn = ['var165','var166','var168','var167','var151','var34','var134']
varssfn = ['U10M','V10M','D2M','T2M','MSL','SSTK','PS']

###

varp = ['var129','var157','var133','var130','var131','var132']
paramP = ['GHT','RH','Q','T','U','V']

for sea, ID in zip(['DJF'],['337315-05']):
    if not os.path.isdir(pi + sea + '-' + ID):
        os.mkdir(pi + sea + '-' + ID)

    pa = sea + '-' + ID + '/'
    
    ### copy original metgrid files into folder to run
    cpp = pi + fol + '/'
    for d in os.listdir(cpp):
        copyfile(cpp+d,pi+pa+d)
    
    pc = pi + pa
    
    Vars = dict()
    nctmp =dict()
    
    ###
    ### 2D fields downloaded
    ###
    sf = 'sf-overlap.nc'

    for va in varname:
        Vars[va] = readcdf(pd + sf,va)[0]	
    for va,par in zip(varsfnn,varssfn):
        Vars[par] = readcdf(pd + sf,va)[0]
    
    lonsoil = readcdf(pd + sf,'lon')
    latsoil = readcdf(pd + sf,'lat')
    ###
    ### 2D fields from thermo era5 data
    ###
    
#    for va in varname2:
#        Vars[va] = readcdf(pd + 'Bmean-' + strength,va)[0]
    
#    lonN = readcdf(pd + 'Bmean-' + strength,'lon')
#    latN = readcdf(pd + 'Bmean-' + strength,'lat')
    
    
    ###
    ### 3D fields
    ###
    
    pl = 'pl-overlap.nc'
    for paraP, va in zip(paramP,varp):
        Vars[paraP] = readcdf(pd + pl,va)[0]
    
    lonP = readcdf(pd + pl,'lon')
    latP = readcdf(pd + pl,'lat')
    
    ###
    ### get right indeces
    ###
        
    losoil = np.where((lonsoil>=doW) & (lonsoil<=doE))[0]
    lasoil = np.where((latsoil<=doN) & (latsoil>=doS))[0]
    
    loP = np.where((lonP>=doW) & (lonP<=doE))[0]
    laP = np.where((latP<=doN) & (latP>=doS))[0]
    
#    loN = np.where((lonN>=doW) & (lonN<=doE))[0]
#    laN = np.where((latN<=doN) & (latN>=doS))[0]
    
    T = Vars['T']
    U = Vars['U']
    V = Vars['V']
    RH = Vars['RH']
    Q = Vars['Q']
    gp = Vars['GHT']/9.81
    
    loS0,loS1,laS0,laS1 = losoil[0],losoil[-1],lasoil[0],lasoil[-1]
    loP0,loP1,laP0,laP1 = loP[0],loP[-1],laP[0],laP[-1]
#    loN0,loN1,laN0,laN1 = loN[0],loN[-1],laN[0],laN[-1]
    
    ### downloaded data starts from N to S hence reverse the index order
    nc = netCDF4.Dataset(pc + 'met_em.d01.2000-12-01_00:00:00.nc',mode='r+')
    ## data is saved in order from 1 hPa to 1000 hPa
    ## so index 0 corresponds to 1 hPa and -1 to 1000 hPa
    ## fill up netcdf from 1000 to 1 hPa thereforerevers order
    for w,p in enumerate(pres):
        nc['RH'][0,-1*(w+1)] = np.flip(RH[w,laP0:laP1,loP0:loP1],axis=0)
        nc['UU'][0,-1*(w+1)] = np.flip(U[w,laP0:laP1,loP0:loP1+1],axis=0)
        nc['VV'][0,-1*(w+1)] = np.flip(V[w,laP0:laP1+1,loP0:loP1],axis=0)
        nc['TT'][0,-1*(w+1)] = np.flip(T[w,laP0:laP1,loP0:loP1],axis=0)
        nc['GHT'][0,-1*(w+1)] =np.flip(gp[w,laP0:laP1,loP0:loP1],axis=0)
    
    nc['TT'][0,0] = np.flip(Vars['T2M'][laS0:laS1,loS0:loS1],axis=0)
    nc['VV'][0,0] = np.flip(Vars['V10M'][laS0:laS1+1,loS0:loS1],axis=0)
    nc['UU'][0,0] = np.flip(Vars['U10M'][laS0:laS1,loS0:loS1+1],axis=0)
    nc['RH'][0,0] = np.flip(100 *( np.exp((17.625 * (Vars['D2M'][laS0:laS1,loS0:loS1]-273.16))/(243.04 + Vars['D2M'][laS0:laS1,loS0:loS1]-273.16))/np.exp((17.625*(nc['TT'][0,0]-273.16))/(243.04 + nc['TT'][0,0]-273.16))),axis=0)
    
    
    nc['PMSL'][0] = np.flip(Vars['MSL'][laS0:laS1,loS0:loS1],axis=0)
    nc['SST'][0] = np.flip(Vars['SSTK'][laS0:laS1,loS0:loS1],axis=0)
    nc['PSFC'][0] =np.flip( Vars['PS'][laS0:laS1,loS0:loS1],axis=0)
    nc['SKINTEMP'][0] = np.flip(Vars['var235'][laS0:laS1,loS0:loS1],axis=0)
    
    ### soil fields have different shape than skintemperature
    for par,va in zip(parameter[1:],varname[1:]):
        print(par,va)
        nc[par][0] = np.flip(Vars[va][0,laS0:laS1,loS0:loS1],axis=0)
    
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

