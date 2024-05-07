import numpy as np
import pandas as pd
import pickle

tracks = '/atmosdyn/michaesp/mincl.era-5/tracks/'

data = pd.read_csv('/atmosdyn2/ascherrmann/011-all-ERA5/data/pandas-basic-data-all-deep-over-sea-12h.csv')
SLP = []
tracktime = []
## 'NA' is saved as nan....
data['reg'].values[3056:26818] = 'NA'

for r in np.unique(data['reg'].values):
    t = dict()
    ID = data['ID'].values[np.where(data['reg'].values==r)[0]]
    dates = data['dates'].values[np.where(data['reg'].values==r)[0]]
    for i,d in zip(ID,dates):
        t[i] = dict()
        y=d[:4]
        m=d[4:6]
        tr = np.loadtxt(tracks + 'fi_' + y + m,skiprows=4)
        if not np.any(tr[:,-1]==i):
            m='%02d'%(int(m)-1)
            if int(m)<1:
                y='%d'%(int(y)-1)
                m='12'
            tr = np.loadtxt(tracks + 'fi_' + y + m,skiprows=4)

        t[i]['SLP'] = tr[np.where(tr[:,-1]==i),3]
        t[i]['reltime'] = tr[np.where(tr[:,-1]==i),0]-tr[np.where(tr[:,-1]==i),0][0][np.where(t[i]['SLP']==np.min(t[i]['SLP']))[0][-1]]


    f = open('/atmosdyn2/ascherrmann/011-all-ERA5/data/' + r + '-SLP-evol-cyclones.txt','rb')
    pickle.dump(t,f)
    f.close()
