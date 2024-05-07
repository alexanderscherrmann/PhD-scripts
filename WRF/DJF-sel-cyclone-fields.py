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
VAR = ['PV300hPa']#,'U300hPa','cycfreq','TH850','omega950','omega900','omega850','omega800','lprecip-06h','lprecip-12h','lprecip-18h','lprecip-24h','cprecip-06h','cprecip-12h','cprecip-18h','cprecip-24h','wcbascfreq','wcbout500freq','wcbout400freq','SLPcycfreq','omega500','MSL']

roie = [7.5,20,35,47]
rois = [6,12,40,50]

for sea in seasons[:1]:
    save[sea] = dict()
    for wi in which[-1:]:
      save[sea][wi] = dict()
      sel = pd.read_csv(ps + sea + '-' + wi)
      newdf = pd.DataFrame(columns=sel.columns)
#
#      #use the ll deepest cyclones
      for ll in [50]:#, 100, 150, 200]:
        save[sea][wi][ll] = dict()
        selp = sel.iloc[:ll]
        for q,los,las,lo,la in zip(range(len(selp['lon'].values)),selp['startlon'].values,selp['startlat'].values,selp['lon'].values,selp['lat'].values):
            if (los>=rois[0] and los<=rois[1] and las>=rois[2] and las<=rois[3] and lo>=roie[0] and lo<=roie[1] and la>=roie[2] and la<=roie[3]):
                newdf = newdf.append(selp.iloc[q],ignore_index=True)

newdf.to_csv(ps + 'selected-intense-cyclones.csv',index=False)


for sea in seasons[:1]:
#    save[sea] = dict()
    for wi in which[-1:]:
#      save[sea][wi] = dict()
      sel = pd.read_csv(ps + 'selected-intense-cyclones.csv')
        ### calc average

      for ll in [50]:
        for we in when:
            save[sea][wi][ll][we] = dict() 
            for var in VAR:
                save[sea][wi][ll][we][var] = np.zeros((len(lats),len(lons)))

            c = 0

            for q,d in enumerate(sel[we].values):
#            for q,d,lo,la in zip(range(len(selp[we].values)),selp[we].values,selp['lon'].values,selp['lat'].values):
                ep = era5 + d[:4] + '/' + d[4:6] + '/'
#                if (lo>=roi[0] and lo<=roi[1] and la>=roi[2] and la<=roi[3]):
                   
                c+=1
                S = ds(ep + 'S' + d,mode='r')

                PV = S.variables['PV'][0,:,la0:la1,lo0:lo1]
                PS = S.variables['PS'][0,la0:la1,lo0:lo1]
                hyam=S.variables['hyam']  # 137 levels  #für G-file ohne levels bis
                hybm=S.variables['hybm']  #   
                ak=hyam[hyam.shape[0]-98:] # only 98 levs are used:
                bk=hybm[hybm.shape[0]-98:]
                
                ps3d=np.tile(PS[:,:],(len(ak),1,1))
                Pr=(ak/100.+bk*ps3d.T).T
                pv300hpa = intp(PV,Pr,300,meta=False)
                save[sea][wi][ll][we]['PV300hPa']+=pv300hpa

            for var in VAR[:]:
                save[sea][wi][ll][we][var]/=c


f = open(ps + 'DJF-intense-selected-PV-2.txt','wb')
pickle.dump(save,f)
f.close()


