import numpy as np
import pickle
import os

p = '/home/ascherrmann/009-ERA-5/MED/'

PV = dict()
count = dict()
locPV = dict()
loccount = dict()

for m in range(0,13):
    PV[m] = 0
    count[m] = 0
    locPV[m] = dict()
    loccount[m] = dict()

    for k in np.arange(-5,42.1,0.5):
        for l in np.arange(30,48.1,0.5):
            locPV[m]['%.1f,%.1f'%(k,l)] = 0
            loccount[m]['%.1f,%.1f'%(k,l)] = 0

for d in os.listdir(p):
    if d.startswith('climatologyPV-w-cyclones-'):
        f = open(p + d,'rb')
        tmp = pickle.load(f)
        f.close()
        for m in range(0,13):
            PV[m] += tmp['PV'][m]
            count[m] += tmp['count'][m]
            for k in tmp['locPV'][m].keys():
                locPV[m][k] += tmp['locPV'][m][k]
                loccount[m][k] += tmp['loccount'][m][k]

f = open(p + 'climatologyPV-w-cyclones.txt','wb')
data = dict()
data['PV'] = PV
data['count'] = count
data['locPV'] = locPV
data['loccount'] = loccount

pickle.dump(data,f)
f.close()
