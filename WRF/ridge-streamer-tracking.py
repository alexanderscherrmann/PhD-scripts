from netCDF4 import Dataset as ds
import numpy as np
import os
import pandas as pd
import pickle 

import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper

minlon = -10
minlat = 30
maxlat = 50
maxlon = 45

minrlon = -70
minrlat = 30
maxrlat = 75
maxrlon = 10

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

lonsr = np.where((LON>=minrlon) & (LON<=maxrlon))[0]
latsr = np.where((LAT<=maxrlat) & (LAT>=minrlat))[0]

lor0,lor1,lar0,lar1 = lonsr[0],lonsr[-1]+1,latsr[0],latsr[-1]+1

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/'
pc = '/atmosdyn/michaesp/mincl.era-5/cdf.final/'

overlap = dict()

refridge = ds(ps + '325832/300/ridge-mask.nc','r')
refridge = refridge.variables['mask'][56,lar0:lar1,lor0:lor1]

refstream = ds(ps + '325832/300/old-mask.nc','r')
refstream = refstream.variables['mask'][56,la0:la1,lo0:lo1]

rrl = len(np.where(refridge==4992)[0])
rsl = len(np.where(refstream==3777)[0])

checkval = 0.9
minn = 400
counters = 0
for dirs in os.listdir(ps):
  if dirs[-1]!='c' and dirs[-1]!='t':

   ID = dirs +'/'
   overlap[int(dirs)] = dict()
#   if ID!='325832/':
#       continue

   for q,pr in enumerate(['300']):
    overlap[int(dirs)][pr] = dict()
    overlap[int(dirs)][pr]['streamer'] = []
    overlap[int(dirs)][pr]['ridge'] = []

    d = ds(ps + ID + '%s/streamer-mask.nc'%pr,'r')
    r = ds(ps + ID + '%s/ridge-mask.nc'%pr,'r')

    mask = d.variables['mask'][:,la0:la1,lo0:lo1]
    rmask = r.variables['mask'][:,lar0:lar1,lor0:lor1]
    prvl = 0
    for k in range(mask.shape[0]):
        for v in np.unique(mask[k])[1:]:
            if v==0:
                continue
            sov = np.zeros_like(refstream)
            sov[refstream==3777] +=1

            sov[mask[k]==v]+=1
            if np.any(sov==2):

                sl = len(np.where(mask[k]==v)[0])
                if sl<checkval:
                    continue
                if sl>=rsl:
                    if len(refstream[np.where(sov==2)])/rsl>checkval:
                        overlap[int(dirs)][pr]['streamer'].append((k,v))
                        #overlap[int(dirs)][pr]['streamer'] = np.append(overlap[int(dirs)][pr]['streamer'],(k,v))
                else:
                    if len(mask[k][np.where(sov==2)])/sl>checkval:
                        overlap[int(dirs)][pr]['streamer'].append((k,v))
                        #overlap[int(dirs)][pr]['streamer'] = np.append(overlap[int(dirs)][pr]['streamer'],(k,v))


        if len(overlap[int(dirs)][pr]['streamer'])==0 or len(overlap[int(dirs)][pr]['streamer'])==prvl:
            continue
        
        if overlap[int(dirs)][pr]['streamer'][-1][0]!=k:
        #    overlap[int(dirs)][pr]['streamer'].remove(overlap[int(dirs)][pr]['streamer'][-1])
            continue

        for v in np.unique(rmask[k][1:]):
            if v==0:
                continue
            rov = np.zeros_like(refridge)
            rov[refridge==4992] +=1
            rov[rmask[k]==v]+=1
            if np.any(rov==2):
                rl = len(np.where(rmask[k]==v)[0])
                if rl>=rrl:
                    if len(refridge[np.where(rov==2)])/rrl>checkval:
                        overlap[int(dirs)][pr]['ridge'].append((k,v))
                        #overlap[int(dirs)][pr]['ridge'] = np.append(overlap[int(dirs)][pr]['ridge'],(k,v))
                else:
                    if len(rmask[k][np.where(rov==2)])/rl>checkval:
                        #overlap[int(dirs)][pr]['ridge'] = np.append(overlap[int(dirs)][pr]['ridge'],(k,v))
                        overlap[int(dirs)][pr]['ridge'].append((k,v))
        prvl = len(overlap[int(dirs)][pr]['streamer'])


    rvls = np.array([])
    for rvl in overlap[int(dirs)][pr]['ridge']:
        rvls = np.append(rvls,rvl[0])
    rvls = np.unique(rvls)

    rmvl = []
    for svl in overlap[int(dirs)][pr]['streamer']:
        if not np.any(rvls==svl[0]):
            rmvl.append(svl)

    for rmv in rmvl:
        overlap[int(dirs)][pr]['streamer'].remove(rmv)
    
    counters+=1

print(counters)
f = open(ps + 'same-ridge-streamer-%.1f-%d.txt'%(checkval,minn),'wb')
pickle.dump(overlap,f)
f.close()

