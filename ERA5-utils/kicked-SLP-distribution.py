import numpy as np
import pickle
import os
import matplotlib
from matplotlib import pyplot as plt
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import pandas as pd

pload = '/home/ascherrmann/009-ERA-5/MED/ctraj/use/'

df = pd.read_csv('/home/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')
df = df.loc[df['ntraj075']>=200]

SLP = df['minSLP'].values
lon = df['lon'].values
lat = df['lat'].values
ID = df['ID'].values
hourstoSLPmin = df['htminSLP'].values
maturedates = df['date'].values

f = open(pload + 'PV-data-dPSP-100-ZB-800-2-400-correct-distance.txt','rb')
PVdata = pickle.load(f)
f.close()

dipv = PVdata['dipv']
datadi = PVdata['rawdata']
dit = PVdata['dit']

both = np.array([])
adv = np.array([])
cyclonic = np.array([])
environmental = np.array([])



#savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']
#var = []
#
#for u,x in enumerate(savings):
#    f = open(pload[:-10] + pload[-9:-4]  + x + 'furthersel.txt',"rb")
#    var.append(pickle.load(f))
#    f.close()
#
#SLP = var[0]
#lon = var[1]
#lat = var[2]
#ID = var[3]
#hourstoSLPmin = var[4]
#dates = var[5]
#avaID = np.array([])
#for k in range(len(ID)):
#    avaID=np.append(avaID,ID[k][0].astype(int))

adv = np.array([])
cyc = np.array([])
env = np.array([])
c = 'cyc'
e = 'env'

ac = dict()
pressuredi = dict()

PVstart = np.array([])
PVend = np.array([])
ca = 0
ct = 0
slpkh = np.array([])
slpkc = np.array([])
slpr = np.array([])

for ll,k in enumerate(dipv.keys()):
    if np.all(ID!=int(k)):
        continue
    q = np.where(ID==int(k))[0][0]
    d = k
#    if (hourstoSLPmin[q][0]>-6):
#        slpkh = np.append(slpkh,SLP[q][abs(hourstoSLPmin[q][0]).astype(int)])
#        continue

    d = k
    OL = PVdata['rawdata'][d]['OL']    
    pre = PVdata['rawdata'][d]['P']
    PV = PVdata['rawdata'][d]['PV']
    i = np.where(PV[:,0]>=0.75)[0]
    
    pvend = PV[i,0]
    pvstart = PV[i,-1]
    PVstart = np.append(PVstart,pvstart)
    PVend = np.append(PVend,pvend)
    
    cypv = dipv[d][c][i,0]
    enpv = dipv[d][e][i,0]
    cy = np.mean(cypv)

#    if (cy<0.12):
#        ca+=1
#        slpkc = np.append(slpkc,SLP[q][abs(hourstoSLPmin[q][0]).astype(int)])
#        continue

    adv = np.append(adv,(pvstart)/pvend)
    cyc = np.append(cyc,cypv/pvend)
    env = np.append(env,enpv/pvend)
    ct +=1
    slpr = np.append(slpr,SLP[q])#[abs(hourstoSLPmin[q][0]).astype(int)])

fig,ax = plt.subplots()

ax.set_ylabel(r'number of cyclones')
ax.set_xlabel(r'minimum SLP [hPa]')
slm = 970
slma = 1010
#ax.hist(slpkh,bins=32,range=[slm,slma],facecolor='k',alpha=0.5)
#ax.hist(slpkc,bins=32,range=[slm,slma],facecolor='r',alpha=0.5)
da = ax.hist(slpr,bins=32,range=[slm,slma],facecolor='grey',edgecolor='k',alpha=1)
ax.set_xlim(970,1010)
ax2 = ax.twinx()
ax2.set_ylim(0,3000)
print(len(np.cumsum(da[0])))
print(len(da[1]))
print(np.cumsum(da[0]))

ax2.plot(da[1],np.append(0,np.cumsum(da[0])),color='k')
ax2.set_ylabel('number of cyclones')
ax.text(0.03, 0.95, 'b)', transform=ax.transAxes,fontsize=12, fontweight='bold',va='top')
fig.savefig('/home/ascherrmann/009-ERA-5/MED/SLP-distribution-EAR5.png',dpi=300,bbox_inches="tight")
plt.close('all')
