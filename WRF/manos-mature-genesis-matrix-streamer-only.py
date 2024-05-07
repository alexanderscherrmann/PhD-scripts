from netCDF4 import Dataset as ds
import numpy as np
import os
import pandas as pd
import pickle

import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper

minlon = -10
minlat = 25
maxlat = 60
maxlon = 50

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

setto = 'genesis'
setto = 'mature'

if setto=='genesis':
    ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/4regionsPV/'
if setto=='mature':
    ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/'

pc = '/atmosdyn/michaesp/mincl.era-5/cdf.final/'

overlap = dict()
streamertypes = dict()

df = pd.read_csv('/atmosdyn2/ascherrmann/011-all-ERA5/data/pandas-basic-data-all-deep-over-sea-12h.csv')
dID = df['ID'].values
htminSLP = df['htSLPmin'].values
mdates = df['dates'].values
months = df['months'].values

sizecheck=1./3.
checkval=0.95
fraction = 'third'

pr = '300'
pool = np.array([])
counts = 0
for dirs in os.listdir(ps):
    if dirs[-1]=='c' or dirs[-1]=='t':
        continue

    if setto=='genesis':
        for kk in range(32,45):
            counts+=1
            pool = np.append(pool,'%06d,%02d'%(int(dirs),kk))
    if setto=='mature':
        for kk in range(40,51):
            counts+=1
            pool = np.append(pool,'%06d,%02d'%(int(dirs),kk))

overlapmeasure = np.zeros((counts,counts))

for q,p in enumerate(pool):
    dirs = p[:6]
    kk = int(p[-2:])
    
    ### define new type of streamer to check for overlap
    ### adjust according to streamer that overlaps with cyclone
    ### and largest ridge in that area

    refs = ds(ps + dirs + '/300/streamer-mask.nc','r')
    refs = refs.variables['mask'][kk,la0:la1,lo0:lo1]

    ####
    #### new streamer and ridge masks
    #### slabel and rlabel
    ####

    slens = np.array([])
    slabels = np.array([])
    for v in np.unique(refs):
      if v==0:
          continue
      slens = np.append(slens,len(np.where(refs==v)[0]))
      slabels = np.append(slabels,v)

    if slens.size==0:
        continue
    ### new lengths of streamer and ridge masks
    ### select largest ridge in that area
    ###
    rsl = len(np.where(refs==slabels[np.argmax(slens)])[0])    

    slabel = slabels[np.argmax(slens)]

    currenttype='%06d,%02d,%d'%(int(dirs),kk,slabel)
    overlap[currenttype] = dict()
    overlap[currenttype]['streamer'] = []
    streamertypes[currenttype] = [currenttype]
    sov = np.zeros_like(refs)
    sov[refs==slabel] +=1
    
    for qq,pp in enumerate(pool):
        if qq<q:
            continue
        if qq==q:
            overlapmeasure[q,qq] = 2
            continue

        dirs=pp[:6]
        k = int(pp[-2:])

        ID = dirs + '/'
        d = ds(ps + ID + '%s/streamer-mask.nc'%pr,'r')
        mask = d.variables['mask'][k,la0:la1,lo0:lo1]
        
        potsov = []
        lens = np.array([])
        streamoverlapmeasure = []

        for v in np.unique(mask):
            if v==0:
                continue

            sov[mask==v]+=1
            sl = len(np.where(mask==v)[0])
            if np.any(sov==2):
              ## check for similar size
                if rsl/sl > sizecheck and sl/rsl>sizecheck:
                    if sl>=rsl:
                        if len(refs[np.where(sov==2)])/rsl>checkval:
                            potsov.append((int(dirs),k,v))
                            lens = np.append(lens,len(np.where(sov==2)[0]))
                            streamoverlapmeasure.append(len(refs[np.where(sov==2)])/rsl)
                    else:
                        if len(mask[np.where(sov==2)])/sl>checkval:
                            potsov.append((int(dirs),k,v))
                            lens = np.append(lens,len(np.where(sov==2)[0]))
                            streamoverlapmeasure.append(len(mask[np.where(sov==2)])/sl)
            
            sov[mask==v]-=1

        if len(potsov)==0:
            continue

        spos = np.argmax(lens)
        overlap[currenttype]['streamer'].append(potsov[spos])
        streamertypes[currenttype].append('%06d,%02d,%d'%(int(dirs),k,potsov[spos][2]))
        overlapmeasure[q,qq] = streamoverlapmeasure[spos]

overlap['matrix'] = overlapmeasure
overlap['pool'] = pool

f = open(ps + setto + '-full-overlap-matrix-%.1f-%s-streamer-only-MED.txt'%(checkval,fraction),'wb')
pickle.dump(overlap,f)
f.close()

f = open(ps + setto + '-full-overlap-test-%.1f-%s-streamer-only-MED.txt'%(checkval,fraction),'wb')
pickle.dump(streamertypes,f)
f.close()
