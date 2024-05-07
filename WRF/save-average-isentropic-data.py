import pickle
import numpy as np
import os

avdata=dict()

f = open('/home/ascherrmann/scripts/WRF/isentropic-data/txtfiles/19790101_00.txt','rb')
d = pickle.load(f)
f.close()

for k in d['2D'].keys():
    avdata[k] = np.zeros(d['2D'][k].shape)

for k in list(d['isentropes'].keys())[1:]:
    avdata[k] = np.zeros(d['isentropes'][k].shape)

TH = d['isentropes']['TH']
i='isentropes'
c2d = 0
counter = np.zeros(d[i]['P'].shape)

for l in os.listdir('/home/ascherrmann/scripts/WRF/isentropic-data/txtfiles/'):
    f = open('/home/ascherrmann/scripts/WRF/isentropic-data/txtfiles/' + l,'rb')
    d = pickle.load(f)
    f.close()

    for k in d['2D'].keys():
        avdata[k] += d['2D'][k]

    c2d +=1
    d[i]['P'][d[i]['P']>1100] = np.NaN

    ma = ~np.isnan(d[i]['P'])
    c = ma.astype(int)
    counter+=c

    for k in list(d[i].keys())[1:]:
        avdata[k][ma] += d[i][k][ma]

avdata['counter']=counter
avdata['2Dcounter']=c2d

f = open('/home/ascherrmann/scripts/WRF/isentropic-data/isentropic-average.txt','wb')
pickle.dump(avdata,f)
f.close()
