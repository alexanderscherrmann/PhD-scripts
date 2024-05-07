# coding: utf-8
import numpy as np
import pickle
SLP = []
lon = []
lat = []
dates = []
hourstoSLPmin = []
ID = []
savings = ['SLP-','lon-','lat-','ID-','hourstoSLPmin-','dates-']
tmp = dict()
fin = dict()

import matplotlib
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt

p = '/home/ascherrmann/009-ERA-5/MED/'

pltlat = np.linspace(0,90,226)[70:121]
pltlon = np.linspace(-180,180,901)[440:541]

minpltlatc = pltlat[0]
maxpltlatc = pltlat[-1]

minpltlonc = pltlon[0]
maxpltlonc = pltlon[-1]


fin = dict()
for x in savings:
    fin[x] = []
for k in range(1979,2021):
  tmp = dict()
  for x in savings:
    f = open(p + x + str(k) + '.txt',"rb")
    tmp[x] = pickle.load(f)
    f.close()
  for l in range(len(tmp['ID-'])):
      if tmp['ID-']==524945:
          for x in savings:
                fin[x].append(tmp[x][l])
          
fin
p = '/home/ascherrmann/011-all-ERA5/data/'
trackpath = '/atmosdyn/michaesp/mincl.era-5/tracks/'

tracks = np.array([])
for d in os.listdir(trackpath):
    if(d.startswith('fi_')):
        if(d[-1]!='s'):
            tracks = np.append(tracks,d)

tracks = np.sort(tracks)
import os
SLP = []
lon = []
lat = []
dates = []
hourstoSLPmin = []
ID = []

savings = ['SLP-','lon-','lat-','ID-','hourstoSLPmin-','dates-','REG-']
yeartmp = 1979

for qqq,fdate in enumerate(tracks[346:]):
    year = int(fdate[-6:-2])
    if year==2018:
        d = np.loadtxt(trackpath + fdate,skiprows=4)
        if np.any(d[:,-1]==524945):
            ids = np.where(d[:,-1]==524945)[0]
            ID.append(d[ids[0],-1])
            SLP.append(np.array(d[ids,3]))
            lon.append(np.array(d[ids,1]))
            lat.append(np.array(d[ids,2]))
            hourstoSLPmin.append(np.array(d[ids,0]-d[np.where(d[ids,3]==np.min(d[ids,3]))[0]]))
            tmp = np.array([])
            for u,v in enumerate(d[ids,0]):
             # if on first days of next month
             if((np.floor(v/24).astype(int) + 1)>days):
                 #if next month is next year
                 if ((int(fdate[-2:]) + 1)>12):
                    #                           add 1 to the year    # so next month -12 in the next year
                    tmp = np.append(tmp,str(int(fdate[-6:-2]) + 1) + '%02d'%(int(fdate[-2:])+1-12) +
                            #days=30, v/24 = 30.5 -> 30-30 + 1          # add the hours of next day
                            '%02d'%(np.floor(v/24).astype(int)-days+1) + '_%02d'%(v%24).astype(int))
                 else:
                     #                  same year               # next month
                    tmp = np.append(tmp,fdate[-6:-2] + '%02d'%(int(fdate[-2:])+1) +
                            #  same as above
                         '%02d'%(np.floor(v/24).astype(int)-days + 1) + '_%02d'%(v%24).astype(int))


             else:
                 tmp = np.append(tmp,str(fdate[-6:]) + '%02d'%(np.floor(v/24).astype(int)+1) +
                         '_%02d'%(v%24).astype(int))
            dates.append(tmp)
            
ID
d = np.loadtxt(trackpath + 'fi_201809',skiprow=4)
d = np.loadtxt(trackpath + 'fi_201809',skiprows=4)
np.where(d[:,-1]==524945)
np.any(d[:,-1]==524945)
if np.any(d[:,-1]==524945):
            ids = np.where(d[:,-1]==524945)[0]
            ID.append(d[ids[0],-1])
            SLP.append(np.array(d[ids,3]))
            lon.append(np.array(d[ids,1]))
            lat.append(np.array(d[ids,2]))
            hourstoSLPmin.append(np.array(d[ids,0]-d[np.where(d[ids,3]==np.min(d[ids,3]))[0]]))
            tmp = np.array([])
            for u,v in enumerate(d[ids,0]):
             # if on first days of next month
             if((np.floor(v/24).astype(int) + 1)>days):
                 #if next month is next year
                 if ((int(fdate[-2:]) + 1)>12):
                    #                           add 1 to the year    # so next month -12 in the next year
                    tmp = np.append(tmp,str(int(fdate[-6:-2]) + 1) + '%02d'%(int(fdate[-2:])+1-12) +
                            #days=30, v/24 = 30.5 -> 30-30 + 1          # add the hours of next day
                            '%02d'%(np.floor(v/24).astype(int)-days+1) + '_%02d'%(v%24).astype(int))
                 else:
                     #                  same year               # next month
                    tmp = np.append(tmp,fdate[-6:-2] + '%02d'%(int(fdate[-2:])+1) +
                            #  same as above
                         '%02d'%(np.floor(v/24).astype(int)-days + 1) + '_%02d'%(v%24).astype(int))


             else:
                 tmp = np.append(tmp,str(fdate[-6:]) + '%02d'%(np.floor(v/24).astype(int)+1) +
                         '_%02d'%(v%24).astype(int))
            dates.append(tmp)
            
hourstoSLPmin
hourstoSLPmin.append(np.array(d[ids,0]-d[np.where(d[ids,3]==np.min(d[ids,3]))[0],0]))
for u,v in enumerate(d[ids,0]):
    if((np.floor(v/24).astype(int) + 1)>days):
                  #if next month is next year
                  if ((int(fdate[-2:]) + 1)>12):
                     #                           add 1 to the year    # so next month -12 in the next year
                     tmp = np.append(tmp,str(int(fdate[-6:-2]) + 1) + '%02d'%(int(fdate[-2:])+1-12) +
                             #days=30, v/24 = 30.5 -> 30-30 + 1          # add the hours of next day
                             '%02d'%(np.floor(v/24).astype(int)-days+1) + '_%02d'%(v%24).astype(int))
                  else:
                      #                  same year               # next month
                     tmp = np.append(tmp,fdate[-6:-2] + '%02d'%(int(fdate[-2:])+1) +
                             #  same as above
                          '%02d'%(np.floor(v/24).astype(int)-days + 1) + '_%02d'%(v%24).astype(int))


    else:
                  tmp = np.append(tmp,str(fdate[-6:]) + '%02d'%(np.floor(v/24).astype(int)+1) +
                          '_%02d'%(v%24).astype(int))
dates.append(tmp)
days=30
for u,v in enumerate(d[ids,0]):
    if((np.floor(v/24).astype(int) + 1)>days):
                  #if next month is next year
                  if ((int(fdate[-2:]) + 1)>12):
                     #                           add 1 to the year    # so next month -12 in the next year
                     tmp = np.append(tmp,str(int(fdate[-6:-2]) + 1) + '%02d'%(int(fdate[-2:])+1-12) +
                             #days=30, v/24 = 30.5 -> 30-30 + 1          # add the hours of next day
                             '%02d'%(np.floor(v/24).astype(int)-days+1) + '_%02d'%(v%24).astype(int))
                  else:
                      #                  same year               # next month
                     tmp = np.append(tmp,fdate[-6:-2] + '%02d'%(int(fdate[-2:])+1) +
                             #  same as above
                          '%02d'%(np.floor(v/24).astype(int)-days + 1) + '_%02d'%(v%24).astype(int))


    else:
                  tmp = np.append(tmp,str(fdate[-6:]) + '%02d'%(np.floor(v/24).astype(int)+1) +
                          '_%02d'%(v%24).astype(int))
dates.append(tmp)
fdate='fi_201809'
for u,v in enumerate(d[ids,0]):
    if((np.floor(v/24).astype(int) + 1)>days):
                  #if next month is next year
                  if ((int(fdate[-2:]) + 1)>12):
                     #                           add 1 to the year    # so next month -12 in the next year
                     tmp = np.append(tmp,str(int(fdate[-6:-2]) + 1) + '%02d'%(int(fdate[-2:])+1-12) +
                             #days=30, v/24 = 30.5 -> 30-30 + 1          # add the hours of next day
                             '%02d'%(np.floor(v/24).astype(int)-days+1) + '_%02d'%(v%24).astype(int))
                  else:
                      #                  same year               # next month
                     tmp = np.append(tmp,fdate[-6:-2] + '%02d'%(int(fdate[-2:])+1) +
                             #  same as above
                          '%02d'%(np.floor(v/24).astype(int)-days + 1) + '_%02d'%(v%24).astype(int))


    else:
                  tmp = np.append(tmp,str(fdate[-6:]) + '%02d'%(np.floor(v/24).astype(int)+1) +
                          '_%02d'%(v%24).astype(int))
dates.append(tmp)
dates
len(dates)
tmp
len(SLP)
SLP
len(lon)
lon
lat
hourstoSLPmin
hourstoSLPmin = []
hourstoSLPmin.append(np.arange(0,len(SLP[0]))-np.arange(0,len(SLP[0]))[np.where(SLP[0]==np.min(SLP[0]))[0]])
hourstoSLPmin
d[ids,0]
dates=[]
for u,v in enumerate(d[ids,0]):
    if((np.floor(v/24).astype(int) + 1)>days):
                  #if next month is next year
                  if ((int(fdate[-2:]) + 1)>12):
                     #                           add 1 to the year    # so next month -12 in the next year
                     tmp = np.append(tmp,str(int(fdate[-6:-2]) + 1) + '%02d'%(int(fdate[-2:])+1-12) +
                             #days=30, v/24 = 30.5 -> 30-30 + 1          # add the hours of next day
                             '%02d'%(np.floor(v/24).astype(int)-days+1) + '_%02d'%(v%24).astype(int))
                  else:
                      #                  same year               # next month
                     tmp = np.append(tmp,fdate[-6:-2] + '%02d'%(int(fdate[-2:])+1) +
                             #  same as above
                          '%02d'%(np.floor(v/24).astype(int)-days + 1) + '_%02d'%(v%24).astype(int))


    else:
                  tmp = np.append(tmp,str(fdate[-6:]) + '%02d'%(np.floor(v/24).astype(int)+1) +
                          '_%02d'%(v%24).astype(int))
tmp=np.array([])
for u,v in enumerate(d[ids,0]):
    if((np.floor(v/24).astype(int) + 1)>days):
                  #if next month is next year
                  if ((int(fdate[-2:]) + 1)>12):
                     #                           add 1 to the year    # so next month -12 in the next year
                     tmp = np.append(tmp,str(int(fdate[-6:-2]) + 1) + '%02d'%(int(fdate[-2:])+1-12) +
                             #days=30, v/24 = 30.5 -> 30-30 + 1          # add the hours of next day
                             '%02d'%(np.floor(v/24).astype(int)-days+1) + '_%02d'%(v%24).astype(int))
                  else:
                      #                  same year               # next month
                     tmp = np.append(tmp,fdate[-6:-2] + '%02d'%(int(fdate[-2:])+1) +
                             #  same as above
                          '%02d'%(np.floor(v/24).astype(int)-days + 1) + '_%02d'%(v%24).astype(int))


    else:
                  tmp = np.append(tmp,str(fdate[-6:]) + '%02d'%(np.floor(v/24).astype(int)+1) +
                          '_%02d'%(v%24).astype(int))
tmp
dates.append(tmp)
zorb = dict()
zorb['SLP'] = SLP[0]
zorb['lon'] = lon[0]
zorb['lat'] = lat[0]
zorb['ID'] = ID[0]
zorb['hourstoSLPmin'] = hourstoSLPmin[0]
zrob['dates'] = dates[0]
zorb['dates'] = dates[0]
f = open('/home/ascherrmann/009-ERA-5/MED/cases/zorbas-data.txt','wb')
pickle.dump(zorb,f)
f.close()
%save -r /home/ascherrmann/scripts/ERA5-utils/zorbas-get-data.py 1-999
