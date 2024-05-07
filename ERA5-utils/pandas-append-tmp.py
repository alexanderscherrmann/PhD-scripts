# coding: utf-8
import numpy as np
import pandas as pd
import pickle
f= open('/home/ascherrmann/009-ERA-5/MED/ctraj/use/PV-data-dPSP-100-ZB-800-2-400-correct-distance.txt',\'rb')
f= open('/home/ascherrmann/009-ERA-5/MED/ctraj/use/PV-data-dPSP-100-ZB-800-2-400-correct-distance.txt','rb')
PVdata = pickle.load(f)
f.close()
datadi = PVdata['rawdata']
df = pd.read_csv('/home/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')
df = df.loc[df['ntraj075']>=200]
ID = df['ID'].values()
ID = df['ID'].values
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
lon = df['lon'].values
lat = df['lat'].values
df = pd.read_csv('/home/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')
ID = df['ID'].values
lon = df['lon'].values
lat = df['lat'].values

avdis = np.zeros(len(ID))
avpres = np.zeros(len(ID))

for q,date in enumerate(datadi.keys()):
    if np.all(ID!=int(date)):
        continue
    I = np.where(ID==int(date))[0][0]
    ids = np.where(datadi[date]['PV'][:,0]>=0.75)[0]
    tmplo = datadi[date]['lon'][ids,0] * (-1)
    tmpla = datadi[date]['lat'][ids,0] * (-1)
    dlo = tmplo + lon[I]
    dla = tmpla + lat[I]
    avdis[I] = np.mean(helper.convert_dlon_dlat_to_radial_dis_new(dlo,dla,lat[I]))
    avpres[I] = np.mean(datadi[date]['P'][ids,0])
    
df['averagepressure'] = avpres
df['averagedistance'] = avdis
df.to_csv('/home/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv',index=False)
%save -r /home/ascherrmann/scripts/ERA5-utils/pandas-append-tmp.py 1-999
