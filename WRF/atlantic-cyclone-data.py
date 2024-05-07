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
tracks = '/atmosdyn/michaesp/mincl.era-5/tracks/'
wcbmask = '/atmosdyn/katih/PhD/data/Gridding/grid_ERA5_r05_100_hit/'


when = ['fourdaypriormature','fivedaypriormature','sixdaypriormature','sevendaypriormature','threedaypriormature','twodaypriormature','onedaypriormature','dates']


### time since first track point of the cyclones as another reference date
#when2 = ['fourdaypriortrack0','fivedaypriortrack0','sixdaypriortrack0','sevendaypriortrack0','threedaypriortrack0','twodaypriortrack0','onedaypriortrack0','track0']


which = ['weak-cyclones.csv','moderate-cyclones.csv','intense-cyclones.csv']
minlon = -80
minlat = 10
maxlat = 70
maxlon = 0

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)
lonshort = np.linspace(-80,0,161)
latshort = np.linspace(10,70,121)
lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1


save = dict()
seasons = ['DJF','MAM','JJA','SON']

for sea in seasons:
    save[sea] = dict()
    for wi in which:
      save[sea][wi] = dict()
      sel = pd.read_csv(ps + sea + '-' + wi)

      #use the ll deepest cyclones
      for ll in [50, 100, 150, 200]:
        save[sea][wi][ll] = dict()
        selp = sel.iloc[:ll]
        ### calc average

        for we in when:
            save[sea][wi][ll][we] = dict() 
            save[sea][wi][ll][we]['lifetime'] = np.array([])
            save[sea][wi][ll][we]['age'] = np.array([])
            save[sea][wi][ll][we]['htSLPmin'] = np.array([])
            save[sea][wi][ll][we]['currentSLP'] = np.array([])
            save[sea][wi][ll][we]['lon'] = np.array([])
            save[sea][wi][ll][we]['lat'] = np.array([])
            save[sea][wi][ll][we]['ID'] = np.array([])
            save[sea][wi][ll][we]['currenthtSLPmin'] = np.array([])
            save[sea][wi][ll][we]['maturelon'] = np.array([])
            save[sea][wi][ll][we]['maturelat'] = np.array([])
            save[sea][wi][ll][we]['minSLP']= np.array([])
            save[sea][wi][ll][we]['dSLP12h'] = np.array([])
            save[sea][wi][ll][we]['dSLP6h'] = np.array([])
            for q,d in enumerate(selp[we].values):

                nd = date.toordinal(date(int(d[:4]),int(d[4:6]),int(d[6:8]))) + int(d[-2:])/24
                ep = era5 + d[:4] + '/' + d[4:6] + '/'
                cf = cycmask + d[:4] + '/' + d[4:6] + '/C' + d

                CM = ds(cf,mode='r')
                mask = CM.variables['LABEL'][0,0,la0:la1,lo0:lo1]
                matloc = CM.variables['PMIN'][0,0,la0:la1,lo0:lo1]
                lifetime = CM.variables['LIFETIME'][0,0,la0:la1,lo0:lo1]
                age = CM.variables['AGE'][0,0,la0:la1,lo0:lo1]

                localPmin = np.where(matloc==1)

                lonlocalPmin = lonshort[localPmin[1]]
                latlocalPmin = latshort[localPmin[0]]
                
                if len(lonlocalPmin)>0:
                    for kq in range(len(lonlocalPmin)):
                        if lifetime[localPmin[0][kq],localPmin[1][kq]] >=48:

                            ID = mask[localPmin[0][kq],localPmin[1][kq]]
                            save[sea][wi][ll][we]['ID'] = np.append(save[sea][wi][ll][we]['ID'],ID)
                            save[sea][wi][ll][we]['lifetime'] = np.append(save[sea][wi][ll][we]['lifetime'],lifetime[localPmin[0][kq],localPmin[1][kq]])
                            save[sea][wi][ll][we]['age'] = np.append(save[sea][wi][ll][we]['age'],age[localPmin[0][kq],localPmin[1][kq]])
                            save[sea][wi][ll][we]['lon'] = np.append(save[sea][wi][ll][we]['lon'],lonlocalPmin[kq])
                            save[sea][wi][ll][we]['lat'] = np.append(save[sea][wi][ll][we]['lat'],latlocalPmin[kq])
                            y = d[:4]
                            m = d[4:6]

                            tr = np.loadtxt(tracks + 'fi_' + y + m,skiprows=4)
                            if not np.any(tr[:,-1]==ID):
                                  m='%02d'%(int(m)-1)
                                  if int(m)<1:
                                    y='%d'%(int(y)-1)
                                    m='12'
                                  tr = np.loadtxt(tracks + 'fi_' + y + m,skiprows=4)
                            ids = tr[:,-1]
                            trid = np.where(ids==ID)[0]
                            slps = tr[trid,3]
                            time = tr[trid,0]
                            lons = tr[trid,1]
                            lats = tr[trid,2]

                            save[sea][wi][ll][we]['currentSLP'] = np.append(save[sea][wi][ll][we]['currentSLP'],slps[save[sea][wi][ll][we]['age'][-1].astype(int)-1])
                            if len(slps)>=(save[sea][wi][ll][we]['age'][-1].astype(int)-1+12):
                                save[sea][wi][ll][we]['dSLP12h'] = np.append(save[sea][wi][ll][we]['dSLP12h'],slps[save[sea][wi][ll][we]['age'][-1].astype(int)-1]-slps[save[sea][wi][ll][we]['age'][-1].astype(int)-1+12])
                            else:
                                save[sea][wi][ll][we]['dSLP12h'] = np.append(save[sea][wi][ll][we]['dSLP12h'],slps[save[sea][wi][ll][we]['age'][-1].astype(int)-1]-slps[save[sea][wi][ll][we]['age'][-1].astype(int)-1-12])

                            if len(slps)>=(save[sea][wi][ll][we]['age'][-1].astype(int)-1+6):
                                save[sea][wi][ll][we]['dSLP6h'] = np.append(save[sea][wi][ll][we]['dSLP6h'],slps[save[sea][wi][ll][we]['age'][-1].astype(int)-1]-slps[save[sea][wi][ll][we]['age'][-1].astype(int)-1+6])
                            else:
                                save[sea][wi][ll][we]['dSLP6h'] = np.append(save[sea][wi][ll][we]['dSLP6h'],slps[save[sea][wi][ll][we]['age'][-1].astype(int)-1]-slps[save[sea][wi][ll][we]['age'][-1].astype(int)-1-6])
                            save[sea][wi][ll][we]['minSLP'] = np.append(save[sea][wi][ll][we]['minSLP'],slps[np.where(slps==np.min(slps))[0][0]])

                            save[sea][wi][ll][we]['htSLPmin'] = np.append(save[sea][wi][ll][we]['htSLPmin'],abs(time[0]-time[np.where(slps==np.min(slps))[0][0]]))
                            save[sea][wi][ll][we]['maturelon'] = np.append(save[sea][wi][ll][we]['maturelon'],lats[np.where(slps==np.min(slps))[0][0]])
                            save[sea][wi][ll][we]['maturelat'] = np.append(save[sea][wi][ll][we]['maturelat'],lons[np.where(slps==np.min(slps))[0][0]])
                            save[sea][wi][ll][we]['currenthtSLPmin'] = np.append(save[sea][wi][ll][we]['currenthtSLPmin'],time[save[sea][wi][ll][we]['age'][-1].astype(int)-1]-time[np.where(slps==np.min(slps))[0][0]])



###
###     here the same thing for the reference of the first track time of the cyclone
###
        
#        for weo,we in zip(when,when2):
#            save[sea][wi][ll][we] = dict()
#            save[sea][wi][ll][we]['lifetime'] = np.array([])
#            save[sea][wi][ll][we]['age'] = np.array([])
#            save[sea][wi][ll][we]['htSLPmin'] = np.array([])
#            save[sea][wi][ll][we]['currentSLP'] = np.array([])
#            save[sea][wi][ll][we]['lon'] = np.array([])
#            save[sea][wi][ll][we]['lat'] = np.array([])
#            save[sea][wi][ll][we]['ID'] = np.array([])
#            save[sea][wi][ll][we]['currenthtSLPmin'] = np.array([])
#            save[sea][wi][ll][we]['maturelon'] = np.array([])
#            save[sea][wi][ll][we]['maturelat'] = np.array([])
#            save[sea][wi][ll][we]['minSLP']= np.array([])
#
#            for q,htslp,d in zip(range(len(selp[weo].values)),selp['htSLPmin'].values,selp[weo].values):
#                nd = date.toordinal(date(int(d[:4]),int(d[4:6]),int(d[6:8]))) + (int(d[-2:])-htslp)/24
#                w = str(helper.datenum_to_datetime(nd))
#                d = w[0:4]+w[5:7]+w[8:10]+'_'+w[11:13]
#
#                ep = era5 + d[:4] + '/' + d[4:6] + '/'
#                cf = cycmask + d[:4] + '/' + d[4:6] + '/C' + d
#
#                CM = ds(cf,mode='r')
#                mask = CM.variables['LABEL'][0,0,la0:la1,lo0:lo1]
#                matloc = CM.variables['PMIN'][0,0,la0:la1,lo0:lo1]
#                lifetime = CM.variables['LIFETIME'][0,0,la0:la1,lo0:lo1]
#                age = CM.variables['AGE'][0,0,la0:la1,lo0:lo1]
#
#                localPmin = np.where(matloc==1)
#
#                lonlocalPmin = lonshort[localPmin[1]]
#                latlocalPmin = latshort[localPmin[0]]
#
#                if len(lonlocalPmin)>0:
#                    for kq in range(len(lonlocalPmin)):
#                        if lifetime[localPmin[0][kq],localPmin[1][kq]] >=48:
#
#                            ID = mask[localPmin[0][kq],localPmin[1][kq]]
#                            save[sea][wi][ll][we]['ID'] = np.append(save[sea][wi][ll][we]['ID'],ID)
#                            save[sea][wi][ll][we]['lifetime'] = np.append(save[sea][wi][ll][we]['lifetime'],lifetime[localPmin[0][kq],localPmin[1][kq]])
#                            save[sea][wi][ll][we]['age'] = np.append(save[sea][wi][ll][we]['age'],age[localPmin[0][kq],localPmin[1][kq]])
#                            save[sea][wi][ll][we]['lon'] = np.append(save[sea][wi][ll][we]['lon'],lonlocalPmin[kq])
#                            save[sea][wi][ll][we]['lat'] = np.append(save[sea][wi][ll][we]['lat'],latlocalPmin[kq])
#                            y = d[:4]
#                            m = d[4:6]
#
#                            tr = np.loadtxt(tracks + 'fi_' + y + m,skiprows=4)
#                            if not np.any(tr[:,-1]==ID):
#                                  m='%02d'%(int(m)-1)
#                                  if int(m)<1:
#                                    y='%d'%(int(y)-1)
#                                    m='12'
#                                  tr = np.loadtxt(tracks + 'fi_' + y + m,skiprows=4)
#                            ids = tr[:,-1]
#                            trid = np.where(ids==ID)[0]
#                            slps = tr[trid,3]
#                            time = tr[trid,0]
#                            lons = tr[trid,1]
#                            lats = tr[trid,2]
#                            
#                            save[sea][wi][ll][we]['currentSLP'] = np.append(save[sea][wi][ll][we]['currentSLP'],slps[save[sea][wi][ll][we]['age'][-1].astype(int)-1])
#
#                            save[sea][wi][ll][we]['htSLPmin'] = np.append(save[sea][wi][ll][we]['htSLPmin'],abs(time[0]-time[np.where(slps==np.min(slps))[0][0]]))
#                            save[sea][wi][ll][we]['minSLP'] = np.append(save[sea][wi][ll][we]['minSLP'],slps[np.where(slps==np.min(slps))[0][0]])
#                            save[sea][wi][ll][we]['maturelon'] = np.append(save[sea][wi][ll][we]['maturelon'],lats[np.where(slps==np.min(slps))[0][0]])
#                            save[sea][wi][ll][we]['maturelat'] = np.append(save[sea][wi][ll][we]['maturelat'],lons[np.where(slps==np.min(slps))[0][0]])
#
#                            save[sea][wi][ll][we]['currenthtSLPmin'] = np.append(save[sea][wi][ll][we]['currenthtSLPmin'],time[save[sea][wi][ll][we]['age'][-1].astype(int)-1]-time[np.where(slps==np.min(slps))[0][0]])


f = open(ps + 'Atlantic-cyclones-prior-intense-moderate-weak-MED-cyclones.txt','wb')
pickle.dump(save,f)
f.close()


