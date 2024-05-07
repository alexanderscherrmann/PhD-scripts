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

which = ['intense-cyclones.csv']
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
seasons = ['DJF']

variables = ['lifetime','age','htSLPmin','currentSLP','lon','lat','ID','currenthtSLPmin','maturelon','maturelat','minSLP','dSLP12h','dSLP6h']

for sea in seasons:
    save[sea] = dict()
    for wi in which:
      save[sea][wi] = dict()
      sel = pd.read_csv(ps + sea + '-' + wi)

      #use the ll deepest cyclones
      for ll in [200]:
        save[sea][wi][ll] = dict()

        for we in when:
            save[sea][wi][ll][we] = dict()
            for clus in np.unique(sel['region'].values):
                selp = sel.iloc[sel['region'].values==clus]
                save[sea][wi][ll][we][clus]=dict()
                for var in variables:
                    save[sea][wi][ll][we][clus][var] = np.array([])

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
                                save[sea][wi][ll][we][clus]['ID'] = np.append(save[sea][wi][ll][we][clus]['ID'],ID)
                                save[sea][wi][ll][we][clus]['lifetime'] = np.append(save[sea][wi][ll][we][clus]['lifetime'],lifetime[localPmin[0][kq],localPmin[1][kq]])
                                save[sea][wi][ll][we][clus]['age'] = np.append(save[sea][wi][ll][we][clus]['age'],age[localPmin[0][kq],localPmin[1][kq]])
                                save[sea][wi][ll][we][clus]['lon'] = np.append(save[sea][wi][ll][we][clus]['lon'],lonlocalPmin[kq])
                                save[sea][wi][ll][we][clus]['lat'] = np.append(save[sea][wi][ll][we][clus]['lat'],latlocalPmin[kq])
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

                                save[sea][wi][ll][we][clus]['currentSLP'] = np.append(save[sea][wi][ll][we][clus]['currentSLP'],slps[save[sea][wi][ll][we][clus]['age'][-1].astype(int)-1])
                                if len(slps)>(save[sea][wi][ll][we][clus]['age'][-1].astype(int)-1+12):
                                    save[sea][wi][ll][we][clus]['dSLP12h'] = np.append(save[sea][wi][ll][we][clus]['dSLP12h'],slps[save[sea][wi][ll][we][clus]['age'][-1].astype(int)-1]-slps[save[sea][wi][ll][we][clus]['age'][-1].astype(int)-1+12])
                                else:
                                    save[sea][wi][ll][we][clus]['dSLP12h'] = np.append(save[sea][wi][ll][we][clus]['dSLP12h'],slps[save[sea][wi][ll][we][clus]['age'][-1].astype(int)-1]-slps[save[sea][wi][ll][we][clus]['age'][-1].astype(int)-1-12])

                                if len(slps)>(save[sea][wi][ll][we][clus]['age'][-1].astype(int)-1+6):
                                    save[sea][wi][ll][we][clus]['dSLP6h'] = np.append(save[sea][wi][ll][we][clus]['dSLP6h'],slps[save[sea][wi][ll][we][clus]['age'][-1].astype(int)-1]-slps[save[sea][wi][ll][we][clus]['age'][-1].astype(int)-1+6])
                                else:
                                    save[sea][wi][ll][we][clus]['dSLP6h'] = np.append(save[sea][wi][ll][we][clus]['dSLP6h'],slps[save[sea][wi][ll][we][clus]['age'][-1].astype(int)-1]-slps[save[sea][wi][ll][we][clus]['age'][-1].astype(int)-1-6])
                                save[sea][wi][ll][we][clus]['minSLP'] = np.append(save[sea][wi][ll][we][clus]['minSLP'],slps[np.where(slps==np.min(slps))[0][0]])

                                save[sea][wi][ll][we][clus]['htSLPmin'] = np.append(save[sea][wi][ll][we][clus]['htSLPmin'],abs(time[0]-time[np.where(slps==np.min(slps))[0][0]]))
                                save[sea][wi][ll][we][clus]['maturelon'] = np.append(save[sea][wi][ll][we][clus]['maturelon'],lats[np.where(slps==np.min(slps))[0][0]])
                                save[sea][wi][ll][we][clus]['maturelat'] = np.append(save[sea][wi][ll][we][clus]['maturelat'],lons[np.where(slps==np.min(slps))[0][0]])
                                save[sea][wi][ll][we][clus]['currenthtSLPmin'] = np.append(save[sea][wi][ll][we][clus]['currenthtSLPmin'],time[save[sea][wi][ll][we][clus]['age'][-1].astype(int)-1]-time[np.where(slps==np.min(slps))[0][0]])


f = open(ps + 'Atlantic-cyclones-cluster.txt','wb')
pickle.dump(save,f)
f.close()


