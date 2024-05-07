import numpy as np
import pandas as pd
from datetime import date

dp = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
days = np.loadtxt(dp + 'WR-ERAI.txt',usecols=1,dtype='str',skiprows=29)
hours = np.loadtxt(dp + 'WR-ERAI.txt',skiprows=29)[:,0].astype(int)
WR = np.loadtxt(dp + 'WR-ERAI.txt',skiprows=29)[:,2].astype(int)

dref = date.toordinal(date(1979,1,1))

when = ['fourdaypriormature','fivedaypriormature','sixdaypriormature','sevendaypriormature','threedaypriormature','twodaypriormature','onedaypriormature','dates']
which = ['weak-cyclones.csv','intense-cyclones.csv']

seasons = ['DJF']#,'MAM','JJA','SON']
for sea in seasons:
    for wi in which:
        sel = pd.read_csv(dp + sea + '-' + wi)
        for we in when:
            tmpWR = np.array([])
            for d in sel[we].values:
                h = int(d[-2:])
                hs = h + (date.toordinal(date(int(d[:4]),int(d[4:6]),int(d[6:8])))-dref) * 24
                loc = np.where(abs(hours-hs)==np.min(abs(hours-hs)))[0][0]
                if np.min(abs(hours-hs))<10:
                    tmpWR = np.append(tmpWR,WR[loc])
                else:
                    tmpWR = np.append(tmpWR,0)
            sel[we + '-WR'] = tmpWR
        sel.to_csv(dp + sea + '-' + wi,index=False)


