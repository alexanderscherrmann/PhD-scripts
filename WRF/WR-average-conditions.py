import numpy as np
import pickle
from datetime import date

dp = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
days = np.loadtxt(dp + 'WR-ERAI.txt',usecols=1,dtype='str',skiprows=29)
hours = np.loadtxt(dp + 'WR-ERAI.txt',skiprows=29)[:,0].astype(int)
WR = np.loadtxt(dp + 'WR-ERAI.txt',skiprows=29)[:,2].astype(int)
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'

### these are the mos prominent ones for intense cyclones
x = np.arange(0,8)
regimes = ['no','AT','GL','AR','ZOEA','ZOWE','BL','ZO']

daydi = dict()

for X,r in zip(x,regimes):    

    ids = np.where(WR==X)[0]
    tmp = np.array([])
    tmpd = days[ids[0]]

    dtmp = date.toordinal(date(int(tmpd[:4]),int(tmpd[4:6]),int(tmpd[6:8])))*24 + int(tmpd[9:])

    tmp= np.append(tmp,tmpd)

    for i in ids:
        da = days[i]
        dah = date.toordinal(date(int(da[:4]),int(da[4:6]),int(da[6:8])))*24 + int(da[9:])
        ddiff = dah-dtmp
        if ddiff >=24:
            tmpd = days[i]
            dtmp = date.toordinal(date(int(tmpd[:4]),int(tmpd[4:6]),int(tmpd[6:8])))*24 + int(tmpd[9:])
            tmp = np.append(tmp,tmpd)

    daydi[r] = tmp
    
f = open(dp + 'WR-days-24h-separated-for-average.txt','wb')
pickle.dump(daydi,f)
f.close()
