import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
rdis=400
pload = '/home/ascherrmann/009-ERA-5/MED/ctraj/use/'
f = open(pload + 'PV-data-dPSP-100-ZB-800-2-%d-correct-distance.txt'%rdis,'rb')
PVdata = pickle.load(f)
f.close()

df = pd.read_csv('/home/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')
df = df.loc[df['ntraj075']>=200]
datadi = PVdata['rawdata']

lon = df['lon'].values
lat = df['lat'].values

dlon = np.arange(-180,180,0.5)
dlat = np.arange(-90,90,0.5)

counter = np.zeros((len(dlat),len(dlon)))

fig,ax = plt.subplots()
for k,lo,la in zip(df['ID'].values,lon,lat):
    cyID = '%06d'%k

    idp = np.where(datadi[cyID]['PV'][:,0]>=0.75)[0]
    tralon = datadi[cyID]['lon'][idp,0]-lo
    tralat = datadi[cyID]['lat'][idp,0]-la

    for tlo, tla in zip(tralon,tralat):
        i = np.where(abs(dlon-tlo)==np.min(abs(dlon-tlo)))[0][0]
        l = np.where(abs(dlat-tla)==np.min(abs(dlat-tla)))[0][0]
        counter[l,i]+=1
#    ax.scatter(tralon,tralat,color='k')


levels=np.array([1e3,5e3,1e4,1.5e4,2e4,2.5e4,3e4])
b =ax.contour(dlon,dlat,counter,colors='k',linewidths=0.5,levels=levels)

ax.set_xlim(-3,3)
ax.set_ylim(-3,3)

ax.set_xticks(ticks=np.arange(-3,3.1,0.5))
ax.set_yticks(ticks=np.arange(-3,3.1,0.5))
ax.scatter(0,0,marker='o',s=30,color='r')
ax.grid(True)


plt.clabel(b,inline=True, fontsize=10, fmt='%d',manual=True)
ax.set_xlabel('longitudinal distance from center [$^{\circ}$]')
ax.set_ylabel('latitudinal distance from center [$^{\circ}$]')

fig.savefig('/home/ascherrmann/009-ERA-5/MED/traj-initilization-pos.png',dpi=300,bbox_inches="tight")
plt.close('all')


