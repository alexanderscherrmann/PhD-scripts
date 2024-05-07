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
maxlon = 60

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1
pc = '/atmosdyn/michaesp/mincl.era-5/cdf.final/'

overlap = dict()
streamertypes = dict()

trange = np.arange(-168,49,3)
df = pd.read_csv('/atmosdyn2/ascherrmann/011-all-ERA5/data/pandas-basic-data-all-deep-over-sea-12h.csv')
dID = df['ID'].values
htminSLP = df['htSLPmin'].values
mdates = df['dates'].values
months = df['months'].values

sizecheck=1./3.
checkval=0.9

setto='genesis'
setto='mature'

if setto=='genesis':
    ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/4regionsPV/'
else:
    ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/'

pr = '300'
pool = np.array([])
counts = 0
poolid = np.array([])
streamermasksdi = dict()
ridgemasksdi = dict()
for dirs in os.listdir(ps):
    if dirs[-1]=='c' or dirs[-1]=='t':
        continue
    tmp = ds(ps + dirs + '/300/streamer-mask.nc','r')
    streamermasksdi[dirs] = tmp.variables['mask'][:,la0:la1,lo0:lo1]
    tmp.close()

    if setto=='genesis':
        for kk in range(32,45):
            pool = np.append(pool,'%06d,%02d'%(int(dirs),kk))
            poolid = np.append(poolid,'%06d'%(int(dirs)))
            counts+=1
    else:
        for kk in range(40,51):
            pool = np.append(pool,'%06d,%02d'%(int(dirs),kk))
            poolid = np.append(poolid,'%06d'%(int(dirs)))
            counts+=1

overlapmeasure = np.zeros(counts)
poolsave = np.stack((pool,overlapmeasure),axis=1)

a = 0
while len(pool)>0:
#if True:
    streamov = []
    ridgeov = []
    overlapsm = []
    idov = []
    #if a==0:
    #    rdirs='309341'
    #    kk=40
    #elif a==1:
    #    rdirs='488924'
    #    kk=35
    #else:
    rdirs = pool[0][:6]
    kk = int(pool[0][-2:])
    dirs=rdirs
    #if a==0:
    #    a+=1
    #dirs = '470479'
    #kk = 9
    loco = np.where(pool=='%06d,%02d'%(int(dirs),kk))[0][0]
    poolsave[loco,1] = 2
    pool = np.delete(pool,loco)
    poolid = np.delete(poolid,loco)

    ### define new type of streamer to check for overlap
    ### adjust according to streamer that overlaps with cyclone
    ### and largest ridge in that area

#    refr = ds(ps + dirs + '/300/ridge-mask.nc','r')
#    refs = ds(ps + dirs + '/300/streamer-mask.nc','r')
#    refr = refr.variables['mask'][kk,lar0:lar1,lor0:lor1]
#    refs = refs.variables['mask'][kk,la0:la1,lo0:lo1]
    refs = streamermasksdi[dirs][kk]

    ####
    #### new streamer and ridge masks
    #### slabel and rlabel
    ####

    slens = np.array([])
    slabels = np.array([])
    for v in np.unique(refs)[1:]:
      slens = np.append(slens,len(np.where(refs==v)[0]))
      slabels = np.append(slabels,v)

    ### new lengths of streamer and ridge masks
    ### select largest ridge in that area
    ###
    rsl = len(np.where(refs==slabels[np.argmax(slens)])[0])    

    slabel = slabels[np.argmax(slens)]

    currenttype='%06d,%02d,%d'%(int(rdirs),kk,slabel)
    overlap[currenttype] = dict()
    overlap[currenttype]['streamer'] = []
    streamertypes[currenttype] = [currenttype]
    overlap[currenttype]['ids'] = np.array([rdirs])
    sov = np.zeros_like(refs)
    sov[refs==slabel] +=1
    
    delids = np.array([])
    for ids in np.unique(poolid):
        locs = np.where(poolid==ids)[0]
        for deli, p in enumerate(pool[locs]):
            dirs=p[:6]
            k = int(p[-2:])
    
            ID = dirs + '/'
#            d = ds(ps + ID + '%s/streamer-mask.nc'%pr,'r')
#            r = ds(ps + ID + '%s/ridge-mask.nc'%pr,'r')
#            mask = d.variables['mask'][k,la0:la1,lo0:lo1]
#            rmask = r.variables['mask'][k,lar0:lar1,lor0:lor1]
            mask = streamermasksdi[dirs][k]

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
            deli = np.where(poolid==dirs)[0]
            pool = np.delete(pool,deli)
            poolid = np.delete(poolid,deli)
            overlap[currenttype]['streamer'].append(potsov[spos])
            streamov.append(potsov[spos])
            idov.append(dirs)
            overlapsm.append(streamoverlapmeasure[spos])
            

            streamertypes[currenttype].append('%06d,%02d,%d'%(int(dirs),k,potsov[spos][2]))
            poolsave[deli+1,1] = streamoverlapmeasure[spos]
            break

    if len(streamov)>=10:
        savedi = dict()
        savedi['stream'] = streamov
        savedi['measure'] = overlapsm
        savedi['ids'] = idov
        f = open(ps + setto + '-start-streamer-cluster-%s-%02d-%.1f.txt'%(rdirs,kk,checkval),'wb')
        pickle.dump(savedi,f)
        f.close()

    overlap[currenttype]['savepool'] = poolsave

f = open(ps + setto + '-start-streamer-manos-full-test-%.1f.txt'%checkval,'wb')
pickle.dump(overlap,f)
f.close()


f = open(ps + setto + '-start-streamer-manos-full-test-2-%.1f.txt'%checkval,'wb')
pickle.dump(streamertypes,f)
f.close()

