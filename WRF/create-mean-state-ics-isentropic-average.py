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

fs = ['mean']
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
paths=['ref-mean-medium-isentropic/']

### copy original metgrid files into folder to run
cpp = pi + fol + '/'

for d in os.listdir(cpp):
    for pa in paths:
        copyfile(cpp+d,pi+pa+d)
	
metf = netCDF4.Dataset(pi + pa + d)
varnc = list(metf.variables.keys())[1:]

# domain boundaries
doN, doW, doE, doS = np.max(metf.corner_lats),np.min(metf.corner_lons),np.max(metf.corner_lons),np.min(metf.corner_lats)
metf.close()

Pmet = np.flip(pres)

varname = ['var235','var39','var40','var41','var42','var139','var170','var183','var236']
parameter = ['SKINTEMP','SM000007','SM007028','SM028100','SM100289','ST000007','ST007028','ST028100','ST100289']

varname2 = ['MSL','SSTK','U10M','V10M','D2M','T2M','PS']
varp = ['var129','var157','var133','var130','var131','var132']
paramP = ['GHT','RH','Q','T','U','V']

fil = open('/home/ascherrmann/scripts/WRF/isentropic-average-on-pressure-level-data.txt','rb')
isentropicdata = pickle.load(fil)
fil.close()

for f,r in zip(fs[:],paths[:]):
    pc = pi + r
    fp = 'P' + f
    
    Vars = dict()
    nctmp =dict()
    
    for va in varname:
        Vars[va] = readcdf(pi + 'monthly-average-soil-skintemp.nc',va)[0]	
    
    lonsoil = readcdf(pi + 'monthly-average-soil-skintemp.nc','lon')
    latsoil = readcdf(pi + 'monthly-average-soil-skintemp.nc','lat')
    
    for va in varname2:
        Vars[va] = isentropicdata[va]
    
    lonN = isentropicdata['lon']
    latN = isentropicdata['lat']
    
    ### thats the average Pfile
    for paraP, va in zip(paramP,varp):
        Vars[paraP] = readcdf(pi+'pl' + f + '.nc',va)[0]
    
    lonP = readcdf(pi+'pl' + f + '.nc','lon')
    latP = np.flip(readcdf(pi+'pl' + f + '.nc','lat'))
    
    
    PS = np.loadtxt(pi + fp + '-surface-pressure.txt')
    
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
    Q = Vars['Q']
    gp = Vars['GHT']/9.81
    
    T =  np.flip(Vars['T'],axis=1)
    U =  np.flip(Vars['U'] ,axis=1)
    V =  np.flip(Vars['V'] ,axis=1)
    RH = np.flip(Vars['RH'],axis=1)
    Q =  np.flip(Vars['Q'],axis=1)
    gp = np.flip(Vars['GHT']/9.81,axis=1)
    
    loS0,loS1,laS0,laS1 = losoil[0],losoil[-1],lasoil[0],lasoil[-1]
    loP0,loP1,laP0,laP1 = loP[0],loP[-1],laP[0],laP[-1]
    loN0,loN1,laN0,laN1 = loN[0],loN[-1],laN[0],laN[-1]
    
    ### downloaded data starts from N to S hence reverse the index order
    nc = netCDF4.Dataset(pc + 'met_em.d01.2000-12-01_00:00:00.nc',mode='r+')
    
    ## data is saved in order from 1 hPa to 1000 hPa
    ## so index 0 corresponds to 1 hPa and -1 to 1000 hPa
    ## fill up netcdf from 1000 to 1 hPa thereforerevers order
    
    for w,p in enumerate(Pmet):
        if p>600 or p<70:
            w2 = np.where(pres==p)[0][0]
            nc['RH'][0,(w+1)] =  RH[w2,laP0:laP1,loP0:loP1]
            nc['UU'][0,(w+1)] =  U[w2,laP0:laP1,loP0:loP1+1]
            nc['VV'][0,(w+1)] =  V[w2,laP0:laP1+1,loP0:loP1]
            nc['TT'][0,(w+1)] =  T[w2,laP0:laP1,loP0:loP1]
            nc['GHT'][0,(w+1)] = gp[w2,laP0:laP1,loP0:loP1]
    
        else:
            w2 = np.where(isentropicdata['P']==p)[0][0]
            nc['RH'][0,(w+1)] =  isentropicdata['RH'][w2,laN0:laN1,loN0:loN1]
            nc['UU'][0,(w+1)] =  isentropicdata['U'][w2,laN0:laN1,loN0:loN1+1]
            nc['VV'][0,(w+1)] =  isentropicdata['V'][w2,laN0:laN1+1,loN0:loN1]
            nc['TT'][0,(w+1)] =  isentropicdata['T'][w2,laN0:laN1,loN0:loN1]
            nc['GHT'][0,(w+1)] = isentropicdata['GHT'][w2,laN0:laN1,loN0:loN1]
     
    ###
    ### 2D variables
    ###
    
    nc['TT'][0,0] = Vars['T2M'][laN0:laN1,loN0:loN1]
    nc['VV'][0,0] = Vars['V10M'][laN0:laN1+1,loN0:loN1]
    nc['UU'][0,0] = Vars['U10M'][laN0:laN1,loN0:loN1+1]
    nc['RH'][0,0] = 100 *( np.exp((17.625 * (Vars['D2M'][laN0:laN1,loN0:loN1]-273.16))/(243.04 + Vars['D2M'][laN0:laN1,loN0:loN1]-273.16))/np.exp((17.625*(nc['TT'][0,0]-273.16))/(243.04 + nc['TT'][0,0]-273.16)))
    
    ###
    ### works until here
    ### runs smooth all 15 days
    ### test now SST and SKINTEMP
    ### they work
    ### include PMSL
    ### works aswell, now check final variable: PSFC
    ### problem was PSFC which needs to be in Pa!! rather than hPa
    ### therefore multiply PSFC with 100 and try again
    ### WORK! :)
    ###
    
    nc['PMSL'][0] = Vars['MSL'][laN0:laN1,loN0:loN1]
    nc['SST'][0] = Vars['SSTK'][laN0:laN1,loN0:loN1]
#    nc['PSFC'][0] = PS[laN0:laN1,loN0:loN1] * 100
    nc['PSFC'][0] = Vars['PS'][laN0:laN1,loN0:loN1]
    
    
    nc['SKINTEMP'][0] = np.flip(Vars['var235'][laS0:laS1,loS0:loS1],axis=0)
    ### soil fields have different shape than skintemperature
    for pa,va in zip(parameter[1:],varname[1:]):
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
