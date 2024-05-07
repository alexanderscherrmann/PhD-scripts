import numpy as np
import pandas as pd
from wrf import interplevel as intp
from netCDF4 import Dataset as ds
import pickle 
from datetime import datetime, date, timedelta
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
era5 = '/atmosdyn2/era5/cdf/'

ep2 = '/atmosdyn2/ascherrmann/009-ERA-5/MED/data/'


cycmask = '/atmosdyn/michaesp/mincl.era-5/cdf.final/'
wcbmask = '/atmosdyn/katih/PhD/data/Gridding/grid_ERA5_r05_100_hit/'


when = ['fourdaypriormature','fivedaypriormature','sixdaypriormature','sevendaypriormature','threedaypriormature','twodaypriormature','onedaypriormature','dates']


### time since first track point of the cyclones as another reference date
when2 = ['fourdaypriortrack0','fivedaypriortrack0','sixdaypriortrack0','sevendaypriortrack0','threedaypriortrack0','twodaypriortrack0','onedaypriortrack0','track0']


which = ['weak-cyclones.csv','moderate-cyclones.csv','intense-cyclones.csv']
minlon = -120
minlat = 10
maxlat = 80
maxlon = 80

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)
lonshort = np.linspace(-120,80,401)
latshort = np.linspace(10,80,141)
lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

# average cyclone 

save = dict()
seasons = ['DJF','MAM','JJA','SON']

## 2D fields to be saved
VAR = ['PV300hPa','U300hPa','cycfreq','TH850','omega950','omega900','omega850','omega800','wcbascfreq','wcbout500freq','wcbout400freq','SLPcycfreq','omega500','MSL','THE850','Q850']
wcbmiss = [np.arange(date.toordinal(date(2008,6,5))*24 + 12, date.toordinal(date(2008,6,9))*24 + 6 + 1),
           np.arange(date.toordinal(date(2014,2,2))*24 + 12, date.toordinal(date(2014,2,2))*24 + 18 + 1),
           np.arange(date.toordinal(date(2015,12,25))*24 + 12, date.toordinal(date(2015,12,26))*24 + 0 + 1)]

for sea in seasons[-1:]:
    save[sea] = dict()
    for wi in which[-1:]:
      save[sea][wi] = dict()
      sel = pd.read_csv(ps + sea + '-' + wi)

      #use the ll deepest cyclones
      for ll in [200]:#[50, 100, 150, 200]:
        save[sea][wi][ll] = dict()
        selp = sel.iloc[:ll]
        ### calc average

        for we in when:
            save[sea][wi][ll][we] = dict() 
            for q,d,ID in zip(range(ll),selp[we].values,selp['ID'].values):
                save[sea][wi][ll][we][ID] = dict()
                for var in VAR:
                    save[sea][wi][ll][we][ID][var] = np.zeros((len(lats),len(lons)))

                ep = era5 + d[:4] + '/' + d[4:6] + '/'
                cf = cycmask + d[:4] + '/' + d[4:6] + '/C' + d
                d2 = d[:-2]
                if int(d[-2:])%6==0:
                    d2+=d[-2:]
                elif int(d[-2:])%6<3:
                    d2+='%02d'%(int(d[-2:])-int(d[-2:])%6)
                else:
                    d2+='%02d'%(int(d[-2:])+6-int(d[-2:])%6)
                    if int(d2[-2:])==24:
                        d2 = d2[:6] + '%02d_%02d'%(int(d2[6:8])+1,0)
                m=int(d[4:6])
                y=int(d[:4])
                if (int(m)<8 and int(m)%2==1) or (int(m)>=8 and int(m)%2==0):
                    dc=31
                elif (int(m)==2):
                    dc=28
                    if y%4==0:
                        dc+=1
                else:
                    dc=30
                if int(d2[6:8])>dc:
                    if (int(d2[4:6])+1)>12:
                        d2 = '%d%02d%02d_%02d'%(int(d2[:4])+1,1,1,int(d2[-2:]))
                    else:
                        d2 = d2[:4] + '%02d%02d_%02d'%(int(d2[4:6])+1,1,int(d2[-2:]))

                S = ds(ep + 'S' + d,mode='r')
                P = ds(ep + 'P' + d,mode='r')
                B = ds(ep + 'B' + d,mode='r')
                CM = ds(cf,mode='r')

                mask = CM.variables['LABEL'][0,0,la0:la1,lo0:lo1]
                MSL = B.variables['MSL'][0,la0:la1,lo0:lo1]

                if d2[:4]!='1979':
                    if np.any(wcbmiss[0]-date.toordinal(date(int(d2[:4]),int(d2[4:6]),int(d2[6:8])))*24 + int(d2[-2:])==0):
                        continue
                    if np.any(wcbmiss[1]-date.toordinal(date(int(d2[:4]),int(d2[4:6]),int(d2[6:8])))*24 + int(d2[-2:])==0):
                        continue
                    if np.any(wcbmiss[2]-date.toordinal(date(int(d2[:4]),int(d2[4:6]),int(d2[6:8])))*24 + int(d2[-2:])==0):
                        continue

                    wf = wcbmask + d2[:4] + '/' + d2[4:6] + '/hit_' + d2
                    WM = ds(wf,mode='r')
                    wcbasc = WM.variables['MIDTROP'][0,0,la0:la1,lo0:lo1]
                    wcbout4 = WM.variables['LT400'][0,0,la0:la1,lo0:lo1]
                    wcbout5 = WM.variables['LT500'][0,0,la0:la1,lo0:lo1]
                    save[sea][wi][ll][we][ID]['wcbascfreq'][wcbasc!=0] += 1
                    save[sea][wi][ll][we][ID]['wcbout400freq'][wcbout4!=0] += 1
                    save[sea][wi][ll][we][ID]['wcbout500freq'][wcbout5!=0] += 1

                #Pminposition = CM.variables['PMIN'][0,0,la0:la1,lo0:lo1]

                save[sea][wi][ll][we][ID]['cycfreq'][mask!=0]+=1
                save[sea][wi][ll][we][ID]['SLPcycfreq'][mask!=0]+=MSL[mask!=0]
                save[sea][wi][ll][we][ID]['MSL']+=MSL

                PV = S.variables['PV'][0,:,la0:la1,lo0:lo1]
                PS = S.variables['PS'][0,la0:la1,lo0:lo1]
                TH = S.variables['TH'][0,:,la0:la1,lo0:lo1]
                THE = S.variables['THE'][0,:,la0:la1,lo0:lo1]
                U = P.variables['U'][0,:,la0:la1,lo0:lo1]
                Q = P.variables['Q'][0,:,la0:la1,lo0:lo1]
                OMEGA = P.variables['OMEGA'][0,:,la0:la1,lo0:lo1]
                hyam=P.variables['hyam']  # 137 levels  #für G-file ohne levels bis
                hybm=P.variables['hybm']  #   ''
                ak=hyam[hyam.shape[0]-98:] # only 98 levs are used:
                bk=hybm[hybm.shape[0]-98:]
                
                ps3d=np.tile(PS[:,:],(len(ak),1,1))
                Pr=(ak/100.+bk*ps3d.T).T

                th850hpa = intp(TH,Pr,850,meta=False)
                u300hpa = intp(U,Pr,300,meta=False)
                pv300hpa = intp(PV,Pr,300,meta=False)
                omega950 = intp(OMEGA,Pr,950,meta=False)
                omega900 = intp(OMEGA,Pr,900,meta=False)
                omega850 = intp(OMEGA,Pr,850,meta=False)
                omega800 = intp(OMEGA,Pr,800,meta=False)
                omega500 = intp(OMEGA,Pr,500,meta=False)
                the850 = intp(THE,Pr,850,meta=False)
                q850 = intp(Q,Pr,850,meta=False)

                save[sea][wi][ll][we][ID]['omega950'] += omega950
                save[sea][wi][ll][we][ID]['omega900'] += omega900
                save[sea][wi][ll][we][ID]['omega850'] += omega850
                save[sea][wi][ll][we][ID]['omega800'] += omega800
                save[sea][wi][ll][we][ID]['omega500'] += omega500

                save[sea][wi][ll][we][ID]['THE850'] += the850
                save[sea][wi][ll][we][ID]['Q850'] += q850
                save[sea][wi][ll][we][ID]['TH850'] += th850hpa
                save[sea][wi][ll][we][ID]['PV300hPa'] += pv300hpa
                save[sea][wi][ll][we][ID]['U300hPa'] += u300hpa

f = open(ps + sea + '-individual-fields.txt','wb')
pickle.dump(save,f)
f.close()


