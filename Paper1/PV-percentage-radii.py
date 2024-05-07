import numpy as np
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pandas as pd


pload = '/atmosdyn2/ascherrmann/009-ERA-5/MED/ctraj/use/'
pload2 = '/atmosdyn2/ascherrmann/009-ERA-5/MED/traj/use/'
psave = '/atmosdyn2/ascherrmann/paper/cyc-env-PV/'

traj = np.array([])
for d in os.listdir(pload):
    if(d.startswith('trajectories-mature-')):
            traj = np.append(traj,d)
traj = np.sort(traj)

savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']
var = []

df = pd.read_csv('/atmosdyn2/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')
thresh='1.5PVU'
df = df.loc[df['ntrajgt%s'%thresh]>=200]

SLP = df['minSLP'].values
lon = df['lon'].values
lat = df['lat'].values
ID = df['ID'].values
hourstoSLPmin = df['htminSLP'].values
maturedates = df['date'].values

for u,x in enumerate(savings):
    f = open(pload2[:-4] + x + 'furthersel.txt',"rb")
    var.append(pickle.load(f))
    f.close()
SLP = var[0]
lon = var[1]
lat = var[2]
IDs = var[3]
hourstoSLPmin = var[4]

avaID = np.array([])
for k in range(len(IDs)):
    avaID=np.append(avaID,IDs[k][0].astype(int))

f = open(pload + 'PV-data-dPSP-100-ZB-800-2-400-correct-distance.txt','rb')
PVdata = pickle.load(f)
f.close()

radii = np.arange(100,1100,100)

globalpercentage = np.zeros(len(radii)+1)

env = 'env'
cyc = 'cyc'
split=[cyc,env]
datadi = dict()
H = 48

datadi = PVdata['rawdata']

for idee,rdis in enumerate(radii):
#  deltaLONLAT = helper.convert_radial_distance_to_lon_lat_dis(rdis)
  wql = 0
  meandi = dict()
  dipv = dict() ####splited pv is stored here
  dit = dict() ### 1 and 0s, 1 if traj is close to center at given time, 0 if not
  meandi[env] = dict()
  meandi[cyc] = dict()
  for uyt,date in enumerate(ID):
#    if uyt>2:
#        continue
    q = np.where(avaID==date)[0][0]
    date = '%06d'%date
    if (hourstoSLPmin[q][0]>0):
        continue

    dipv[date]=dict()    #accumulated pv is saved here
    dipv[date][env]=dict()
    dipv[date][cyc]=dict()

    ### follow cyclone backwards to find its center
    dit[date] = dict()
    dit[date][env] = np.zeros(datadi[date]['time'].shape)
    dit[date][cyc] = np.zeros(datadi[date]['time'].shape)

    CLON = np.array([])
    CLAT = np.array([])
    for k in range(0,H+1):
        if(np.any(hourstoSLPmin[q]==(-1 * k))):
            CLON = np.append(CLON,lon[q][np.where(hourstoSLPmin[q]==(-1*k))[0][0]])
            CLAT = np.append(CLAT,lat[q][np.where(hourstoSLPmin[q]==(-1*k))[0][0]])
        else:
            ### use boundary position that no traj should be near it
            CLON = np.append(CLON,10000)
            CLAT = np.append(CLAT,lat[q][0])
    ### check every hours every trajecttory whether it is close to the center ###
    for e, h in enumerate(datadi[date]['time'][0,:]):
        ### 30.10.2020 radial distance instead of square
        ### if traj is in circle of 200km set cyc entry to 0, else env entry 1
        for tr in range(len(datadi[date]['time'])):
            if (helper.convert_dlon_dlat_to_radial_dis_new(CLON[e]-datadi[date]['lon'][tr,e],CLAT[e]-datadi[date]['lat'][tr,e],CLAT[e])<=rdis):
#            if ( np.sqrt( (CLON[e]-datadi[date]['lon'][tr,e])**2 + (CLAT[e]-datadi[date]['lat'][tr,e])**2) <=  deltaLONLAT):
                dit[date][cyc][tr,e]=1
            else:
                dit[date][env][tr,e]=1
    cycID=date
    deltaPV = np.zeros(datadi[cycID]['time'].shape)
    deltaPV[:,1:] = datadi[cycID]['PV'][:,:-1]-datadi[cycID]['PV'][:,1:]
    
    idp = np.where(datadi[cycID]['PV'][:,0]>=0.75)[0]

    for key in split:
        dipv[cycID][key] = np.zeros(datadi[cycID]['time'].shape)
        dipv[cycID][key][:,:-1] = np.flip(np.cumsum(np.flip(abs(deltaPV[:,1:])*dit[cycID][key][:,1:],axis=1),axis=1),axis=1)
        if wql==0:
           meandi[key] = dipv[date][key][idp,0]
        else:
           meandi[key] = np.append(meandi[key], dipv[date][key][idp,0])
    wql=10

  globalpercentage[idee+1] = np.mean(meandi[cyc]/(meandi[cyc] + meandi[env]))
globalpercentage[0] = 0

print(np.append(0,radii))
print(globalpercentage*100)
fig, axes = plt.subplots()
axes.plot(np.append(0,radii),globalpercentage*100)
axes.set_ylabel('relative absolute cyclonic PV modification [%]')
axes.set_xlabel('cyclone effective radius [km]')
axes.set_xticks(ticks=np.arange(0,1100,200))

axes.axvline(400,color='grey',ls='-')
axes.set_xlim(0,1000)
axes.set_ylim(0,40)
name = 'PV-percentage-400-0hours-correct-distance-PV-%s.png'%thresh
fig.savefig(psave + name,dpi=300,bbox_inches="tight")
plt.close('all')

