from netCDF4 import Dataset as ds
import numpy as np
import pandas as pd
import pickle
from wrf import interplevel as intp
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper

trackp = '/atmosdyn/michaesp/mincl.era-5/tracks/'
df = pd.read_csv('/atmosdyn2/ascherrmann/011-all-ERA5/data/pandas-basic-data-all-deep-over-sea-12h.csv')
dID = df['ID'].values
htminSLP = df['htSLPmin'].values
mdates = df['dates'].values

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/'
era5 = '/atmosdyn2/era5/cdf/'
slpf = np.array([])
slph = np.array([])
slpm = np.array([])

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

tmp = ds(era5 + '2000/10/H20001010_10','r')
pH = np.ones_like(tmp.variables['Z'][0,:]) * tmp.variables['plev'][:][:,None,None]/100.
tmp.close()
pH = pH[:,:21,:21]
refloc = np.where(pH[:,0,0]==850)[0][0]

tmp = ds(era5 + '2000/10/S20001010_10','r')
ak = tmp.variables['hyam'][137-98:]
bk = tmp.variables['hybm'][137-98:]
g = 9.81

#p3dh = np.ones((len(press),31,31)) * press[:,None,None]/100.
PVdi = dict()
rdi = dict()
pdi = dict()
gdi = dict()
grdi = dict()
rrefdi = dict()

for l in ['low','upp']:
    PVdi[l] = dict()
    rdi[l] = dict()
    pdi[l] = dict()
    gdi[l] = dict()
    grdi[l] = dict()
    rrefdi[l]  = dict()

for dirs in os.listdir(ps):
  if dirs[-1]!='c' and dirs[-1]!='t':
    for l in ['low','upp']:
        PVdi[l][dirs] = dict()
        rdi[l][dirs] = dict()
        pdi[l][dirs] = dict()
        gdi[l][dirs] = dict()
        grdi[l][dirs] = dict()
        rrefdi[l][dirs] = dict()

    htslp = htminSLP[np.where(dID==int(dirs))[0][0]]
    md = mdates[np.where(dID==int(dirs))[0][0]]
    yyyy = md[:4]
    mm = md[4:6]
    print(md)
    
    slptrack = np.loadtxt(trackp + 'fi_' + yyyy + mm,skiprows=4)
    if not np.any(slptrack[:,-1]==int(dirs)):
        mm = int(mm)-1
        if mm<1:
            mm=12
            yyyy = '%d'%(int(yyyy)-1)

        slptrack = np.loadtxt(trackp + 'fi_' + yyyy + '%02d'%mm,skiprows=4)

    slps = slptrack[np.where(slptrack[:,-1]==int(dirs))[0],3]
    lons = slptrack[np.where(slptrack[:,-1]==int(dirs))[0],1]
    lats = slptrack[np.where(slptrack[:,-1]==int(dirs))[0],2]
    slpf = np.append(slpf,slps[0])
    slpm = np.append(slpm,slps[int(htslp)])
    slph = np.append(slph,slps[int(htslp/2)])

    lonf = lons[0]
    lonm = lons[int(htslp)]
    lonh = lons[int(htslp/2)]

    latf = lats[0]
    latm = lats[int(htslp)]
    lath = lats[int(htslp/2)]

    daf = helper.change_date_by_hours(md,-1 * htslp)
    dam = md
    dah = helper.change_date_by_hours(md,-1* int(htslp/2))

    loa = [lonf,lonh,lonm]
    laa = [latf,lath,latm]
    maa = [daf,dah,dam]

    for lo,la,d,k in zip(loa,laa,maa,['f','h','m']):
        if lo%0.5!=0:
            if lo%0.5<0.25:
                lo-=lo%0.5
            else:
                lo+=(0.5-lo%0.5)
        if la%0.5!=0:
            if la%0.5<0.25:
                la-=la%0.5
            else:
                la+=(0.5-la%0.5)

        los = np.where((LON>=lo-5) & (LON<=lo+5))[0]
        las = np.where((LAT>=la-5) & (LAT<=la+5))[0]
        
        dlo2 = np.ones((21,21)) * np.arange(-5,5.5,0.5)[None,:]
        dla2 = np.ones((21,21)) * np.arange(-5,5.5,0.5)[:,None]
        lat2 = dla2 + la

        R = helper.convert_dlon_dlat_to_radial_dis_new(dlo2,dla2,lat2)

        H = ds(era5 + d[:4] + '/' + d[4:6] + '/H' + d,'r')
        GHT = H.variables['Z'][0,:,las[0]:las[-1]+1,los[0]:los[-1]+1]/g
        H.close()
        
        S = ds(era5 + d[:4] + '/' + d[4:6] + '/S' + d,'r')
        PS = S.variables['PS'][0,las[0]:las[-1]+1,los[0]:los[-1]+1]
        PV = S.variables['PV'][0,:,las[0]:las[-1]+1,los[0]:los[-1]+1]

        p3dS = np.tile(PS,(PV.shape[0],1,1))
        p3dS = (ak/100 + bk * p3dS.T).T

        S.close()
        ###
        ### lower PV anomaly
        ###
        zi,yi,xi = np.where(p3dS>=700)
        grdi['low'][dirs][k] = np.ones(len(zi))*(-500)
        PVdi['low'][dirs][k] = np.ones(len(zi))*(-500)
        rdi['low'][dirs][k] = np.ones(len(zi))*(-500)
        pdi['low'][dirs][k] = np.ones(len(zi))*(-500)
        gdi['low'][dirs][k] = np.ones(len(zi))*(-500)
        rrefdi['low'][dirs][k] = np.ones(len(zi))*(-500)
#
#
#        print(lo,la,GHT.shape,pH.shape)
        for x,y,z,q in zip(xi,yi,zi,range(len(zi))):

            grdi['low'][dirs][k][q] = GHT[refloc,y,x]            
            p = p3dS[z,y,x]
            pv = PV[z,y,x]
            ght = intp(GHT,pH,p,meta=False)[y,x]
            ghtref = GHT[refloc,y,x]
            r = np.sqrt(ght**2 + (R[x,y]*1000)**2)
            rref = np.sqrt((ght-ghtref)**2 + (R[x,y]*1000)**2)
            
            PVdi['low'][dirs][k][q] = pv/r
            rdi['low'][dirs][k][q] =  r
            pdi['low'][dirs][k][q] =  p
            gdi['low'][dirs][k][q] = ght
            rrefdi['low'][dirs][k][q] = rref
#        ###
#        ### upper PV anomaly
#        ### 
#
        zi,yi,xi = np.where((p3dS<=450)& (p3dS>=250))
        grdi['upp'][dirs][k] = np.ones(len(zi))*(-500)
        rrefdi['upp'][dirs][k] = np.ones(len(zi))*(-500)
        PVdi['upp'][dirs][k] = np.ones(len(zi))*(-500)
        rdi['upp'][dirs][k] = np.ones(len(zi))*(-500)
        pdi['upp'][dirs][k] = np.ones(len(zi))*(-500)
        gdi['upp'][dirs][k] = np.ones(len(zi))*(-500)
        for x,y,z,q in zip(xi,yi,zi,range(len(zi))):
            grdi['upp'][dirs][k][q] = GHT[refloc,y,x]
            p = p3dS[z,y,x]
            pv = PV[z,y,x]
            ght = intp(GHT,pH,p,meta=False)[y,x]
            ghtref = GHT[refloc,y,x]
            r = np.sqrt(ght**2 + (R[x,y]*1000)**2)
            rref = np.sqrt((ght-ghtref)**2 + (R[x,y]*1000)**2)

            gdi['upp'][dirs][k][q] = ght
            PVdi['upp'][dirs][k][q] = pv/r
            rdi['upp'][dirs][k][q] =  r
            pdi['upp'][dirs][k][q] =  p        
            rrefdi['upp'][dirs][k][q] = rref

dic = dict()
dic['pv'] = PVdi
dic['r'] = rdi
dic['p'] = pdi
dic['ght'] = gdi
dic['rref'] = rrefdi

#f = open('/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/test-PV-distance.txt','rb')
#dic  = pickle.load(f)
#f.close()

dic['slp'] = np.stack((slpf,slph,slpm),axis=1)
dic['ghtref850'] = grdi

f = open('/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/test-PV-distance.txt','wb')
pickle.dump(dic,f)
f.close()





