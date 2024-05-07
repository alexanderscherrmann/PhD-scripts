import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

p = '/atmosdyn2/ascherrmann/011-all-ERA5/'
ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'

df = pd.read_csv(p + 'data/pandas-basic-data-all-deep-over-sea-12h.csv')

MEDend = np.where(df['reg']=='MED')[0][-1] + 1
r = 'MED'

X = 300

def get_Xdays_before_mature_stage(d,X):

    if (int(d[6:8])-X)>=1:
        return d[:6] + '%02d'%(int(d[6:8])-X) + d[-3:]
    
    monthsn = int(d[4:6])-1
    if monthsn<1:
        return '%d'%(int(d[:4])-1) + '12' + '%02d'%(int(d[6:8]) + 31 - X) + d[-3:]

    if monthsn<8 and monthsn%2==1:
        days = 31
    elif monthsn==2:
        days =28
        if int(d[:4])%4==0:
            days=29
    elif monthsn>=8 and monthsn%2==0:
        days=31
    else:
        days=30
    return d[:4] + '%02d'%(monthsn) + '%02d'%(int(d[6:8]) + days - X) + d[-3:]


columns = df.columns

tmp = df.loc[df['reg']==r]
SLP = tmp['minSLP'].values
mSLP = np.mean(SLP)
# average cyclone ids
sort = np.argsort(abs(SLP-mSLP))[:X]

selm = tmp.iloc[sort]
fdpm= np.array([])
sdpm= np.array([])
svdpm = np.array([])
fodpm= np.array([])
thdpm = np.array([])
tdpm = np.array([])
odpm = np.array([])

for d in selm['dates'].values:
    fdpm = np.append(fdpm,get_Xdays_before_mature_stage(d,5))
    sdpm = np.append(sdpm,get_Xdays_before_mature_stage(d,6))
    svdpm = np.append(svdpm,get_Xdays_before_mature_stage(d,7))
    fodpm = np.append(fodpm,get_Xdays_before_mature_stage(d,4))
    thdpm  = np.append(thdpm,get_Xdays_before_mature_stage(d,3))
    tdpm  = np.append(tdpm,get_Xdays_before_mature_stage(d,2))
    odpm = np.append(odpm,get_Xdays_before_mature_stage(d,1))

selm['fivedaypriormature']=fdpm
selm['sixdaypriormature']=sdpm
selm['sevendaypriormature']=svdpm
selm['fourdaypriormature'] =fodpm
selm['threedaypriormature']=thdpm
selm['twodaypriormature'] = tdpm
selm['onedaypriormature'] = odpm

selm.to_csv(ps + 'moderate-%d-cyclones.csv'%X,index=False)

# weak cyclones
selw = tmp.iloc[::-1][:X]
fdpm= np.array([])
sdpm= np.array([])
svdpm = np.array([])
fodpm= np.array([])
thdpm = np.array([])
tdpm = np.array([])
odpm = np.array([])
for d in selw['dates'].values:
    fdpm = np.append(fdpm,get_Xdays_before_mature_stage(d,5))
    sdpm = np.append(sdpm,get_Xdays_before_mature_stage(d,6))
    svdpm = np.append(svdpm,get_Xdays_before_mature_stage(d,7))
    fodpm = np.append(fodpm,get_Xdays_before_mature_stage(d,4))
    thdpm  = np.append(thdpm,get_Xdays_before_mature_stage(d,3))
    tdpm  = np.append(tdpm,get_Xdays_before_mature_stage(d,2))
    odpm = np.append(odpm,get_Xdays_before_mature_stage(d,1))

selw['fivedaypriormature']=fdpm
selw['sixdaypriormature']=sdpm
selw['sevendaypriormature']=svdpm
selw['fourdaypriormature'] =fodpm
selw['threedaypriormature']=thdpm
selw['twodaypriormature'] = tdpm
selw['onedaypriormature'] = odpm

selw.to_csv(ps + 'weak-%d-cyclones.csv'%X,index=False)


# intense cyclones
seli = tmp.iloc[:X]
fodpm= np.array([])
fdpm= np.array([])
sdpm= np.array([])
svdpm = np.array([])
thdpm = np.array([])
tdpm = np.array([])
odpm = np.array([])

for d in seli['dates'].values:
    fdpm = np.append(fdpm,get_Xdays_before_mature_stage(d,5))
    sdpm = np.append(sdpm,get_Xdays_before_mature_stage(d,6))
    svdpm = np.append(svdpm,get_Xdays_before_mature_stage(d,7))
    fodpm = np.append(fodpm,get_Xdays_before_mature_stage(d,4))
    thdpm  = np.append(thdpm,get_Xdays_before_mature_stage(d,3))
    tdpm  = np.append(tdpm,get_Xdays_before_mature_stage(d,2))
    odpm = np.append(odpm,get_Xdays_before_mature_stage(d,1))

seli['fivedaypriormature']=fdpm
seli['sixdaypriormature']=sdpm
seli['sevendaypriormature']=svdpm
seli['fourdaypriormature'] =fodpm
seli['threedaypriormature']=thdpm
seli['twodaypriormature'] = tdpm
seli['onedaypriormature'] = odpm

seli.to_csv(ps + 'intense-%d-cyclones.csv'%X,index=False)



