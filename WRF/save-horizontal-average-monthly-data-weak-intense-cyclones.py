import numpy as np
import pandas as pd
from wrf import interplevel as intp
from netCDF4 import Dataset as ds
import pickle 
from datetime import datetime, date, timedelta
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import os



ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
era5 = '/atmosdyn2/era5/cdf/'

ep2 = '/atmosdyn2/ascherrmann/009-ERA-5/MED/data/'


cycmask = '/atmosdyn/michaesp/mincl.era-5/cdf.final/'
wcbmask = '/atmosdyn/katih/PhD/data/Gridding/grid_ERA5_r05_100_hit/'


when = ['fourdaypriormature','fivedaypriormature','sixdaypriormature','sevendaypriormature','threedaypriormature','twodaypriormature','onedaypriormature','dates']


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
months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
monthsn = np.arange(1,13)
## 2D fields to be saved
VAR = ['PV300hPa','U300hPa','cycfreq','TH850','omega950','omega900','omega850','omega800','lprecip-24h','cprecip-24h','SLPcycfreq','omega500','MSL','Q850','THE850','Pat1.5PVU','Pat2PVU','wcbascfreq','wcbout500freq','wcbout400freq']

#precidorder = ['precip-06h','precip-12h','precip-18h','precip-24h']
precidorder = ['precip-24h']
wcbmiss = [np.arange(date.toordinal(date(2008,6,5))*24 + 12, date.toordinal(date(2008,6,9))*24 + 6 + 1),
           np.arange(date.toordinal(date(2014,2,2))*24 + 12, date.toordinal(date(2014,2,2))*24 + 18 + 1),
           np.arange(date.toordinal(date(2015,12,25))*24 + 12, date.toordinal(date(2015,12,26))*24 + 0 + 1)]


for mn,mo in zip(monthsn,months):
    save[mo] = dict()
    for wi in which:
      save[mo][wi] = dict()
      sel = pd.read_csv(ps + mo + '-' + wi)

      #use the ll deepest cyclones
      for ll in [50]:
        save[mo][wi][ll] = dict()
        selp = sel.iloc[:ll]
        ### calc average

        for we in when:
            save[mo][wi][ll][we] = dict() 
            for var in VAR:
                save[mo][wi][ll][we][var] = np.zeros((len(lats),len(lons)))

            save[mo][wi][ll][we]['ymature'] = np.array([])
            save[mo][wi][ll][we]['xmature'] = np.array([])
            save[mo][wi][ll][we]['wcbcounter'] = 0


            for q,d in enumerate(selp[we].values):

#                nd = date.toordinal(date(int(d[:4]),int(d[4:6]),int(d[6:8]))) + int(d[-2:])/24
#
#                precid00 = nd*24
#                precid24 = nd*24-23
#
#                precid = [precid24]
#                for pco, prec in zip(precidorder,precid):
#                    dhours = np.arange(prec,precid00+1)
#                    dhours /= 24
#                    for qq,dho in enumerate(dhours):
#                        whours = str(helper.datenum_to_datetime(dho))
#                        precdates = ep2 + 'P' + whours[0:4]+whours[5:7]+whours[8:10]+'_'+whours[11:13]
#                        cmd = 'cdo select,name=LSP,CP ' + precdates + ' /home/ascherrmann/scripts/WRF/tmph%02d'%qq
#                        os.system(cmd)
#                    cmd = 'cdo enssum /home/ascherrmann/scripts/WRF/tmph* /home/ascherrmann/scripts/WRF/tmph'
#                    os.system(cmd)
#                    PREDATA = ds('/home/ascherrmann/scripts/WRF/tmph',mode='r')
#                    save[mo][wi][ll][we]['c'+pco]+=PREDATA.variables['CP'][0,la0:la1,lo0:lo1]
#                    save[mo][wi][ll][we]['l'+pco]+=PREDATA.variables['LSP'][0,la0:la1,lo0:lo1]
#
#                    os.system('rm /home/ascherrmann/scripts/WRF/tmph*')
#
                ep = era5 + d[:4] + '/' + d[4:6] + '/'
                cf = cycmask + d[:4] + '/' + d[4:6] + '/C' + d
#                d2 = d[:-2]
#                if int(d[-2:])%6==0:
#                    d2+=d[-2:]
#                elif int(d[-2:])%6<3:
#                    d2+='%02d'%(int(d[-2:])-int(d[-2:])%6)
#                else:
#                    d2+='%02d'%(int(d[-2:])+6-int(d[-2:])%6)
#                    if int(d2[-2:])==24:
#                        d2 = d2[:6] + '%02d_%02d'%(int(d2[6:8])+1,0)
#                m=int(d[4:6])
#                y=int(d[:4])
#                if (int(m)<8 and int(m)%2==1) or (int(m)>=8 and int(m)%2==0):
#                    dc=31
#                elif (int(m)==2):
#                    dc=28
#                    if y%4==0:
#                        dc+=1
#                else:
#                    dc=30
#                if int(d2[6:8])>dc:
#                    if (int(d2[4:6])+1)>12:
#                        d2 = '%d%02d%02d_%02d'%(int(d2[:4])+1,1,1,int(d2[-2:]))
#                    else:
#                        d2 = d2[:4] + '%02d%02d_%02d'%(int(d2[4:6])+1,1,int(d2[-2:]))

                S = ds(ep + 'S' + d,mode='r')
                P = ds(ep + 'P' + d,mode='r')
                B = ds(ep + 'B' + d,mode='r')
                CM = ds(cf,mode='r')

                mask = CM.variables['LABEL'][0,0,la0:la1,lo0:lo1]
                MSL = B.variables['MSL'][0,la0:la1,lo0:lo1]
                matloc = CM.variables['PMIN'][0,0,la0:la1,lo0:lo1]
                save[mo][wi][ll][we]['xmature'] = np.append(save[mo][wi][ll][we]['xmature'],np.where(matloc!=0)[1])
                save[mo][wi][ll][we]['ymature'] = np.append(save[mo][wi][ll][we]['ymature'],np.where(matloc!=0)[0])

#                if d2[:4]!='1979':
#                    if np.any(wcbmiss[0]-date.toordinal(date(int(d2[:4]),int(d2[4:6]),int(d2[6:8])))*24 + int(d2[-2:])==0):
#                        continue
#                    if np.any(wcbmiss[1]-date.toordinal(date(int(d2[:4]),int(d2[4:6]),int(d2[6:8])))*24 + int(d2[-2:])==0):
#                        continue
#                    if np.any(wcbmiss[2]-date.toordinal(date(int(d2[:4]),int(d2[4:6]),int(d2[6:8])))*24 + int(d2[-2:])==0):
#                        continue
#
#                    wf = wcbmask + d2[:4] + '/' + d2[4:6] + '/hit_' + d2
#                    WM = ds(wf,mode='r')
#                    wcbasc = WM.variables['MIDTROP'][0,0,la0:la1,lo0:lo1]
#                    wcbout4 = WM.variables['LT400'][0,0,la0:la1,lo0:lo1]
#                    wcbout5 = WM.variables['LT500'][0,0,la0:la1,lo0:lo1]
#                    save[mo][wi][ll][we]['wcbascfreq'][wcbasc!=0] += 1
#                    save[mo][wi][ll][we]['wcbout400freq'][wcbout4!=0] += 1
#                    save[mo][wi][ll][we]['wcbout500freq'][wcbout5!=0] += 1
#                    save[mo][wi][ll][we]['wcbcounter']+=1

                #Pminposition = CM.variables['PMIN'][0,0,la0:la1,lo0:lo1]

                save[mo][wi][ll][we]['cycfreq'][mask!=0]+=1
                save[mo][wi][ll][we]['SLPcycfreq'][mask!=0]+=MSL[mask!=0]
                save[mo][wi][ll][we]['MSL']+=MSL

                PV = S.variables['PV'][0,:,la0:la1,lo0:lo1]
                PS = S.variables['PS'][0,la0:la1,lo0:lo1]
                TH = S.variables['TH'][0,:,la0:la1,lo0:lo1]
                Q = P.variables['Q'][0,:,la0:la1,lo0:lo1]
                THE = S.variables['THE'][0,:,la0:la1,lo0:lo1]

                U = P.variables['U'][0,:,la0:la1,lo0:lo1]
                OMEGA = P.variables['OMEGA'][0,:,la0:la1,lo0:lo1]
                hyam=P.variables['hyam']  # 137 levels  #für G-file ohne levels bis
                hybm=P.variables['hybm']  #   ''
                ak=hyam[hyam.shape[0]-98:] # only 98 levs are used:
                bk=hybm[hybm.shape[0]-98:]
                
                ps3d=np.tile(PS[:,:],(len(ak),1,1))
                Pr=(ak/100.+bk*ps3d.T).T
                pr500 = Pr[:56]
                PV500 = PV[:56]
                pres15PVU = intp(pr500,PV500,1.5,meta=False)
                pres20PVU = intp(pr500,PV500,2,meta=False)

#                th850hpa = intp(TH,Pr,850,meta=False)
#                u300hpa = intp(U,Pr,300,meta=False)
#                pv300hpa = intp(PV,Pr,300,meta=False)
#                omega950 = intp(OMEGA,Pr,950,meta=False)
#                omega900 = intp(OMEGA,Pr,900,meta=False)
#                omega850 = intp(OMEGA,Pr,850,meta=False)
#                omega800 = intp(OMEGA,Pr,800,meta=False)
#                omega500 = intp(OMEGA,Pr,500,meta=False)
#
                the850 = intp(THE,Pr,850,meta=False)
                q850 = intp(Q,Pr,850,meta=False)

                save[mo][wi][ll][we]['THE850']+= the850
                save[mo][wi][ll][we]['Q850'] += q850
                save[mo][wi][ll][we]['Pat2PVU'] += pres20PVU
                save[mo][wi][ll][we]['Pat1.5PVU'] +=pres15PVU

#                save[mo][wi][ll][we]['omega950'] += omega950
#                save[mo][wi][ll][we]['omega900'] += omega900
#                save[mo][wi][ll][we]['omega850'] += omega850
#                save[mo][wi][ll][we]['omega800'] += omega800
#                save[mo][wi][ll][we]['omega500'] += omega500
#
#                save[mo][wi][ll][we]['TH850'] += th850hpa
#                save[mo][wi][ll][we]['PV300hPa'] += pv300hpa
#                save[mo][wi][ll][we]['U300hPa'] += u300hpa

            for var in VAR[:-3]:
                save[mo][wi][ll][we][var]/=ll


#            save[mo][wi][ll][we]['wcbascfreq']/=save[mo][wi][ll][we]['wcbcounter']
#            save[mo][wi][ll][we]['wcbout400freq']/=save[mo][wi][ll][we]['wcbcounter']
#            save[mo][wi][ll][we]['wcbout500freq']/=save[mo][wi][ll][we]['wcbcounter']
#
#            save[mo][wi][ll][we]['xmature']=save[mo][wi][ll][we]['xmature'].astype(int)
#            save[mo][wi][ll][we]['ymature']=save[mo][wi][ll][we]['ymature'].astype(int)


    f = open(ps + mo + '-average-fields2.txt','wb')
    pickle.dump(save,f)
    f.close()


