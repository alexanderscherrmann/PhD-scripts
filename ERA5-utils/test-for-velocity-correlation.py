# coding: utf-8
import numpy as np
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm
import argparse
import pickle
pload = '/home/ascherrmann/009-ERA-5/MED/ctraj/use/'
pload2 = '/home/ascherrmann/009-ERA-5/MED/traj/use/'
LON=np.round(np.linspace(-180,180,721),1)
LAT=np.round(np.linspace(-90,90,361),1)

savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']
var = []

for u,x in enumerate(savings):
    f = open(pload2[:-4] + x + 'furthersel.txt',"rb")
    var.append(pickle.load(f))
    f.close()

Slp = var[0]
SLP = var[0]
Clon = var[1]
Clat = var[2]
ID = var[3]
hourstoSLPmin = var[4]
Clon[0]
avaID = np.array([])
maturedates = np.array([])
SLPminid = np.array([])
end = np.array([])
lons = []
lats = []
for k in range(len(ID)):
    avaID=np.append(avaID,ID[k][0].astype(int))
    loc = np.where(hourstoSLPmin[k]==0)[0][0]
    lons.append(np.array(Clon[k][:loc+1]))
    lats.append(np.array(Clat[k][:loc+1]))
    maturedates = np.append(maturedates,dates[k][loc])
    SLPminid = np.append(SLPminid,loc)
    end = np.append(end,maturedates[k] + '-ID-%06d.txt'%avaID[k])
    
avaID = np.array([])
maturedates = np.array([])
SLPminid = np.array([])
end = np.array([])
lons = []
lats = []
for k in range(len(ID)):
    avaID=np.append(avaID,ID[k][0].astype(int))
    loc = np.where(hourstoSLPmin[k]==0)[0][0]
    lons.append(np.array(Clon[k][:loc+1]))
    lats.append(np.array(Clat[k][:loc+1]))
    
    SLPminid = np.append(SLPminid,loc)
    end = np.append(end,maturedates[k] + '-ID-%06d.txt'%avaID[k])
    
avaID = np.array([])
maturedates = np.array([])
SLPminid = np.array([])
end = np.array([])
lons = []
lats = []
for k in range(len(ID)):
    avaID=np.append(avaID,ID[k][0].astype(int))
    loc = np.where(hourstoSLPmin[k]==0)[0][0]
    lons.append(np.array(Clon[k][:loc+1]))
    lats.append(np.array(Clat[k][:loc+1]))
    
    SLPminid = np.append(SLPminid,loc)
    
    
lons
import pandas as pd
df = pd.read_csv('/home/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')
df = df.loc[df['ntraj075']>=200]
ids = df['ID'].values
avaID = np.array([])
maturedates = np.array([])
SLPminid = np.array([])
end = np.array([])
lons = []
lats = []
for k in range(len(ID)):
    if np.all(ids!=ID[k][0]):
        continue
    avaID=np.append(avaID,ID[k][0].astype(int))
    loc = np.where(hourstoSLPmin[k]==0)[0][0]
    lons.append(np.array(Clon[k][:loc+1]))
    lats.append(np.array(Clat[k][:loc+1]))
    
    SLPminid = np.append(SLPminid,loc)
    
    
f = open('/home/ascherrmann/009-ERA-5/MED/ctraj/use/PV-data-dPSP-100-ZB-800-2-400-correct-distance.txt','rb')
data = pickle.load(f)
f.close()
dipv = data['dipv']
dfv = pd.DataFrame(colums=['cycper','envper','advper','v'])
cy = np.array([])
env = np.array([])
adv = np.array([])
v = np.array([])
for k in dipv.keys():
    if np.all(avaID!=int(k)):
        continue
    i = np.where(avaID==int(k))[0][0]
    if len(lons[i])<3:
        continue
    v = np.append(v,np.sqrt((lons[1:]-lons[:-1])**2 + (lats[1:]-lats[:-1])**2))
    cy = np.append(cy,np.mean(dipv[k]['cyc'][:,0]/data['rawdata'][k]['PV'][:,0]))
    env = np.append(env,np.mean(dipv[k]['env'][:,0]/data['rawdata'][k]['PV'][:,0]))
    adv = np.append(adv, np.mean(data['rawdata'][k]['PV'][:,-1]/data['rawdata'][k]['PV'][:,0]))
    
dfv = pd.DataFrame(columns=['cycper','envper','advper','v'])
cy = np.array([])
env = np.array([])
adv = np.array([])
v = np.array([])
for k in dipv.keys():
    if np.all(avaID!=int(k)):
        continue
    i = np.where(avaID==int(k))[0][0]
    if len(lons[i])<3:
        continue
    v = np.append(v,np.sqrt((lons[1:]-lons[:-1])**2 + (lats[1:]-lats[:-1])**2))
    cy = np.append(cy,np.mean(dipv[k]['cyc'][:,0]/data['rawdata'][k]['PV'][:,0]))
    env = np.append(env,np.mean(dipv[k]['env'][:,0]/data['rawdata'][k]['PV'][:,0]))
    adv = np.append(adv, np.mean(data['rawdata'][k]['PV'][:,-1]/data['rawdata'][k]['PV'][:,0]))
    
dfv = pd.DataFrame(columns=['cycper','envper','advper','v'])
cy = np.array([])
env = np.array([])
adv = np.array([])
v = np.array([])
for k in dipv.keys():
    if np.all(avaID!=int(k)):
        continue
    i = np.where(avaID==int(k))[0][0]
    if len(lons[i])<3:
        continue
    v = np.append(v,np.sqrt((np.array(lons[1:])-np.array(lons[:-1]))**2 + (np.array(lats[1:])-np.array(lats[:-1]))**2))
    cy = np.append(cy,np.mean(dipv[k]['cyc'][:,0]/data['rawdata'][k]['PV'][:,0]))
    env = np.append(env,np.mean(dipv[k]['env'][:,0]/data['rawdata'][k]['PV'][:,0]))
    adv = np.append(adv, np.mean(data['rawdata'][k]['PV'][:,-1]/data['rawdata'][k]['PV'][:,0]))
    
dfv = pd.DataFrame(columns=['cycper','envper','advper','v'])
cy = np.array([])
env = np.array([])
adv = np.array([])
v = np.array([])
for k in dipv.keys():
    if np.all(avaID!=int(k)):
        continue
    i = np.where(avaID==int(k))[0][0]
    if len(lons[i])<3:
        continue
    v = np.append(v,np.sqrt((np.array(lons[i][1:])-np.array(lons[i][:-1]))**2 + (np.array(lats[i][1:])-np.array(lats[i][:-1]))**2))
    cy = np.append(cy,np.mean(dipv[k]['cyc'][:,0]/data['rawdata'][k]['PV'][:,0]))
    env = np.append(env,np.mean(dipv[k]['env'][:,0]/data['rawdata'][k]['PV'][:,0]))
    adv = np.append(adv, np.mean(data['rawdata'][k]['PV'][:,-1]/data['rawdata'][k]['PV'][:,0]))
    
dfv['cycper'] = cy
dfv['envper'] = env
dfv['advper'] = adv
dfv['v'] = v
v
len(v)
dfv = pd.DataFrame(columns=['cycper','envper','advper','v'])
cy = np.array([])
env = np.array([])
adv = np.array([])
v = np.array([])
for k in dipv.keys():
    if np.all(avaID!=int(k)):
        continue
    i = np.where(avaID==int(k))[0][0]
    if len(lons[i])<3:
        continue
    v = np.append(v,np.mean(np.sqrt((np.array(lons[i][1:])-np.array(lons[i][:-1]))**2 + (np.array(lats[i][1:])-np.array(lats[i][:-1]))**2)))
    cy = np.append(cy,np.mean(dipv[k]['cyc'][:,0]/data['rawdata'][k]['PV'][:,0]))
    env = np.append(env,np.mean(dipv[k]['env'][:,0]/data['rawdata'][k]['PV'][:,0]))
    adv = np.append(adv, np.mean(data['rawdata'][k]['PV'][:,-1]/data['rawdata'][k]['PV'][:,0]))
    
dfv['v'] = v
dfv['advper'] = adv
dfv['envper'] = env
dfv['cycper'] = cy
dfv
from scipy.stats import pearsonr
pearsonr(cycper,v)
pearsonr(dfv['cycper'],v)
pearsonr(dfv['env'],v)
pearsonr(dfv['envper'],v)
pearsonr(dfv['advper'],v)
%save -r /home/ascherrmann/scripts/ERA5-utils/test-for-velocity-correlation.py 1-999
