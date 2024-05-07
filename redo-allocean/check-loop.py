import pandas as pd
import numpy as np
import pickle
import os

p = '/atmosdyn2/ascherrmann/011-all-ERA5/data/'
trackpath = '/atmosdyn/michaesp/mincl.era-5/tracks/'

regions = ['MED','NA','SA','NP','SP','IO']
savings = ['SLP-','lon-','lat-','ID-','hourstoSLPmin-','dates-','REG-']

add = 'deep-over-sea'
wrongfiles = np.array([])
wronght = np.array([])
wronglon = np.array([])
wronglat = np.array([])
wrongslp = np.array([])
for k in range(1979,1980):
    #load SLP,LON,LAT,ID,etc into tmp
    tmp = dict()
    for x in savings:
        f = open(p + x + str(k) + '-' + add + '.txt',"rb")
        tmp[x] = pickle.load(f)
        f.close()

    for l, ID in enumerate(tmp['ID-']):
        tmplon = np.array(tmp['lon-'][l])
        tmplat = np.array(tmp['lat-'][l])
        tmpslp = np.array(tmp['SLP-'][l])
        tmpht = np.array(tmp['hourstoSLPmin-'][l])

        date = tmp['dates-'][l][0]

        track = np.loadtxt(trackpath+'fi_' + date[:6],skiprows=4)
        ids = track[:,-1]

        if not np.any(ids==ID):
            wrongfiles=np.append(wrongfiles,ID)
            continue

        loc = np.where(ids==ID)[0]
        lon = track[loc,1]
        lat = track[loc,2]
        slp = track[loc,3]
        ht = track[loc,0]-track[loc[np.where(slp==np.min(slp))[0][-1]],0]
        
        compht = ht-tmpht
        if not np.all(compht==0):
            wronght = np.append(wronght,ID)

        complat = lat-tmplat
        if not np.all(complat==0):
            wronglat = np.append(wronglat,ID)

        complon = lon-tmplon
        if not np.all(complon==0):
            wronglon = np.append(wronglon,ID)

        compslp= slp-tmpslp
        if not np.all(compslp==0):
            wrongslp = np.append(wrongslp,ID)


di = dict()
di['date'] = wrongfiles
di['lat'] = wronglat
di['lon'] = wronglon
di['slp'] = wrongslp
di['ht'] = wronght
di['originalIDs'] = np.array(tmp['ID-'])



f = open('/home/ascherrmann/scripts/redo-allocean/loopcheck.txt','wb')
pickle.dump(di,f)
f.close()
