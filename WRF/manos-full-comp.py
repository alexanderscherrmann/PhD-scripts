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
maxlat = 65
maxlon = 60

minrlon = -70
minrlat = 30
maxrlat = 65
maxrlon = 40

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

lonsr = np.where((LON>=minrlon) & (LON<=maxrlon))[0]
latsr = np.where((LAT<=maxrlat) & (LAT>=minrlat))[0]

lor0,lor1,lar0,lar1 = lonsr[0],lonsr[-1]+1,latsr[0],latsr[-1]+1

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/4regionsPV/'
pc = '/atmosdyn/michaesp/mincl.era-5/cdf.final/'

overlap = dict()
streamertypes = dict()

df = pd.read_csv('/atmosdyn2/ascherrmann/011-all-ERA5/data/pandas-basic-data-all-deep-over-sea-12h.csv')
dID = df['ID'].values
htminSLP = df['htSLPmin'].values
mdates = df['dates'].values
months = df['months'].values

sizecheck=0.
checkval=0.8
fraction = 'all'

pr = '300'
pool = np.array([])
counts = 0
for dirs in os.listdir(ps):
    if dirs[-1]=='c' or dirs[-1]=='t':
        continue

    counts +=1
    for kk in range(65):
        pool = np.append(pool,'%06d,%02d'%(int(dirs),kk))

overlapmeasure = np.zeros((counts * 65,counts*65))

for q,p in enumerate(pool[:20]):
    dirs = p[:6]
    kk = int(p[-2:])
    
    ### define new type of streamer to check for overlap
    ### adjust according to streamer that overlaps with cyclone
    ### and largest ridge in that area

    refr = ds(ps + dirs + '/300/ridge-mask.nc','r')
    refs = ds(ps + dirs + '/300/streamer-mask.nc','r')
    refr = refr.variables['mask'][kk,lar0:lar1,lor0:lor1]
    refs = refs.variables['mask'][kk,la0:la1,lo0:lo1]

    ####
    #### new streamer and ridge masks
    #### slabel and rlabel
    ####

    rlabels = np.array([])
    rlens = np.array([])
    for v in np.unique(refr)[1:]:
      rlens = np.append(rlens,len(np.where(refr==v)[0]))
      rlabels = np.append(rlabels,v)

    slens = np.array([])
    slabels = np.array([])
    for v in np.unique(refs)[1:]:
      slens = np.append(slens,len(np.where(refs==v)[0]))
      slabels = np.append(slabels,v)

    ### new lengths of streamer and ridge masks
    ### select largest ridge in that area
    ###
    rrl = len(np.where(refr==rlabels[np.argmax(rlens)])[0])
    rsl = len(np.where(refs==slabels[np.argmax(slens)])[0])    

    rlabel = rlabels[np.argmax(rlens)]
    slabel = slabels[np.argmax(slens)]

    currenttype='%06d,%02d,%d,%d'%(int(dirs),kk,slabel,rlabel)
    overlap[currenttype] = dict()
    overlap[currenttype]['ridge'] = []
    overlap[currenttype]['streamer'] = []
    streamertypes[currenttype] = [currenttype]
    sov = np.zeros_like(refs)
    sov[refs==slabel] +=1
    rov = np.zeros_like(refr)
    rov[refr==rlabel] +=1 
    
    for qq,pp in enumerate(pool[:]):
        if qq<q:
            continue
        if qq==q:
            overlapmeasure[q,qq] = 2
            continue

        dirs=pp[:6]
        k = int(pp[-2:])

        ID = dirs + '/'
        d = ds(ps + ID + '%s/streamer-mask.nc'%pr,'r')
        r = ds(ps + ID + '%s/ridge-mask.nc'%pr,'r')
        mask = d.variables['mask'][k,la0:la1,lo0:lo1]
        rmask = r.variables['mask'][k,lar0:lar1,lor0:lor1]
        
        potsov = []
        potrov = []
        lens = np.array([])
        streamoverlapmeasure = []
        ridgeoverlapmeasure = []

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
        lens = np.array([])
        for v in np.unique(rmask):
            if v==0:
                continue
            rov[rmask==v]+=1
            rl = len(np.where(rmask==v)[0])

            if np.any(rov==2):
                if rrl/rl > sizecheck and rl/rrl>sizecheck:
                    if rl>=rrl:
                        if len(refr[np.where(rov==2)])/rrl>checkval:
                            potrov.append((int(dirs),k,v))
                            lens = np.append(lens,len(np.where(rov==2)[0]))
                            ridgeoverlapmeasure.append(len(refr[np.where(rov==2)])/rrl)
                    else:
                        if len(rmask[np.where(rov==2)])/rl>checkval:
                            potrov.append((int(dirs),k,v))
                            lens = np.append(lens,len(np.where(rov==2)[0]))
                            ridgeoverlapmeasure.append(len(rmask[np.where(rov==2)])/rl)
            
            rov[rmask==v]-=1
       
        if len(potrov)==0:
            continue

        rpos = np.argmax(lens)
        overlap[currenttype]['streamer'].append(potsov[spos])
        overlap[currenttype]['ridge'].append(potrov[rpos])
        streamertypes[currenttype].append('%06d,%02d,%d,%d'%(int(dirs),k,potsov[spos][2],potrov[rpos][2]))
        overlapmeasure[q,qq] = streamoverlapmeasure[spos] + ridgeoverlapmeasure[rpos]


overlap['matrix'] = overlapmeasure
overlap['pool'] = pool

#f = open(ps + 'full-overlap-matrix-%.1f-%s.txt'%(checkval,fraction),'wb')
#pickle.dump(overlap,f)
#f.close()

#f = open(ps + 'full-overlap-test-%.1f-%s.txt'%(checkval,fraction),'wb')
#pickle.dump(streamertypes,f)
#f.close()
