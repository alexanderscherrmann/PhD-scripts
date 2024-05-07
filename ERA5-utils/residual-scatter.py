import numpy as np
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import matplotlib
import matplotlib.pyplot as plt

p = '/home/ascherrmann/009-ERA-5/'
traced = np.array([])
for d in os.listdir(p):
    if(d.startswith('traced-vars-2full')):
            traced = np.append(traced,d)
traced = np.sort(traced)

labs = helper.traced_vars_ERA5()

var='PV'
unit = 'PVU'
#for testing H=47,normally 48
H = 47

fig, axes = plt.subplots()
ax=axes

PVR = np.array(['pvf','pvt'])

resm = np.array([])
datadi = dict()
dipv =dict()
for u, e in enumerate(traced):
    d = np.loadtxt(p + e)
    ID = int(e[-10:-4])
#    date = e[12:23]
    datadi[ID] = dict()
    dipv[ID] = dict()

    for q,el in enumerate(labs):
        datadi[ID][el] = d[:,q].reshape(-1,H+1)

    dipv[ID]['APVTOT'] = np.zeros(datadi[ID]['time'].shape)
    for pvr in PVR:
        dipv[ID][pvr] = np.zeros(datadi[ID]['time'].shape)
        dipv[ID][pvr][:,1:-1] = np.flip(np.cumsum(np.flip(datadi[ID][pvr][:,1:-1],axis=1),axis=1),axis=1)
        dipv[ID]['APVTOT'] += dipv[ID][pvr]

    largePV = np.where(datadi[ID]['PV'][:,0]>0.3)[0]
    lowPV = np.where(datadi[ID]['PV'][:,0]<0.3)[0]

    PVstart = datadi[ID]['PV'][:,-1]
    PVend = datadi[ID]['PV'][:,1]

    res = np.mean(dipv[ID]['APVTOT'][:,0] - (PVend-PVstart))
    lowPVres = np.mean(dipv[ID]['APVTOT'][lowPV,0]- (PVend[lowPV]-PVstart[lowPV]))
    largePVres = np.mean(dipv[ID]['APVTOT'][largePV,0]- (PVend[largePV]-PVstart[largePV]))

#    if abs(res)<10:
    resm = np.append(resm,res)
    ax.scatter(u+1,res)
    ax.scatter(u+1,lowPVres,marker='d')
    ax.scatter(u+1,largePVres,marker='x')
resm[np.where(abs(resm)>1)[0]] = 0.3
ax.set_ylim(-.7,.7)
ax.set_xticks(ticks=np.arange(1,u+2))
ax.axhline(np.mean(resm),color='black',lw=1.0)
ax.set_label('residual')
name='residualsfull.png'
fig.savefig(p + name,dpi=300,bbox_inches="tight")
plt.close()
