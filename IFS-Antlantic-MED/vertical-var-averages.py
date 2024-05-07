import numpy as np
import xarray as xr
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
from matplotlib import cm
import pickle


MONTHS = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
MONTHSN = np.arange(1,13,1)

p = '/home/ascherrmann/010-IFS/traj/MED/use/'
path=p
dates = np.array([])
ids = np.array([])
for d in os.listdir(p):
    if(d.startswith('trajectories-mature-')):
            dates = np.append(dates,d[-25:-14])

dates = np.sort(dates)

MON = np.array([])
for d in os.listdir(p[:-4]):
    if(d.startswith('traend-')):
        MON = np.append(MON,d)
MON = np.sort(MON)

LON=np.linspace(-180,180,901)
LAT=np.linspace(0,90,226)

f = open(p[:-13] + 'data/All-CYC-entire-year-NEW-correct.txt','rb')
PVdata = pickle.load(f)
f.close()

rdis=200

varS = np.array(['PV','TH','THE'])
varP = np.array(['T'])
di = dict()
di2= dict()
tmpd = dict()

relvort = np.array([])

for q,date in enumerate(dates):
    
    di[date] = dict()
    di2[date] = dict()
    mon = MON[q][-9:-4]
    ID = int(MON[q][-16:-10]) 
    ana_path='/home/ascherrmann/010-IFS/data/' + mon + '/'
    mat = np.where(PVdata[mon][ID]['dates']==date)[0][0]

    clat = PVdata[mon][ID]['clat'][mat]
    clon = PVdata[mon][ID]['clon'][mat]
    relvort = np.append(relvort,PVdata[mon][ID]['zeta'][mat])

    sfile = ana_path + 'S' + date
    pfile = ana_path + 'P' + date

    s = xr.open_dataset(sfile, drop_variables=['P','RH','VORT','PVRCONVT','PVRCONVM','PVRTURBT','PVRTURBM','PVRLS','PVRCOND','PVRSW','PVRLWH','PVRLWC','PVRDEP','PVREVC','PVREVR','PVRSUBI','PVRSUBS','PVRMELTI','PVRMELTS','PVRFRZ','PVRRIME','PVRBF'])

    p = xr.open_dataset(pfile,drop_variables=['SWC', 'RWC', 'IWC', 'LWC','PS','CC','OMEGA','tsw','tlw','tmix','tconv','tcond','tdep','tbf','tevc','tsubi','tevr','tsubs','tmelti','tmelts','tfrz','trime','udotconv','vdotconv','udotmix','vdotmix','tls','tce'])

    PS = s.PS.values[0,0,clat,clon]
    for q in varS:
        tmp = getattr(s,q)
        di[date][q] = np.transpose(tmp.values[0,:,clat,clon])
        di2[date][q] = np.array([])

    for q in varP:
        tmp = getattr(p,q)
        di[date][q] = np.transpose(tmp.values[0,:,clat,clon])
        di2[date][q] = np.array([])

    di2[date]['THEstar'] = np.array([])
    for pres in np.arange(100,1001,25):
        for q in np.append(varS,varP):
            tmpd[q] = np.array([])
        for e in range(len(clat)):
            P = helper.modellevel_to_pressure(PS[e])
            I = (np.where(abs(P-pres)==np.min(abs(P-pres)))[0][0]).astype(int)
            for q in np.append(varS,varP):
                tmpd[q] = np.append(tmpd[q],di[date][q][I,e])

        for q in np.append(varS,varP):
                di2[date][q] = np.append(di2[date][q],np.mean(tmpd[q]))
        di2[date]['THEstar'] = np.append(di2[date]['THEstar'],np.mean(helper.theta_star(tmpd['TH'],tmpd['T'],pres)))


pltvar = np.append(varS,'THEstar')
col = np.array(['grey','red','midnightblue'])
fig, ax = plt.subplots(1,4,sharey=True,figsize=(10,6))
for k, date in enumerate(dates):
    for w,q in enumerate(pltvar[:]):
        ax[w].plot(di2[date][q],np.arange(100,1001,25),color='grey',linewidth=2)#cm.bwr((relvort[k]-np.min(relvort))/(np.max(relvort)-np.min(relvort))),linewidth=2)


lvls = dict()
av = dict()
xlims = np.array([[-1,10],[280,420],[280,420],[280,420]])
xlab = np.array(['PV [PVU]',r'$\theta$ [K]',r'$\theta_e$ [K]',r'$\theta_e^{*}$ [K]'])
for q,pres in enumerate(np.arange(100,1001,25)):
    lvls[pres] = dict()
    for var in np.append(varS,'THEstar'):
        av[var] = np.array([])
        lvls[pres][var] = np.array([])
        for date in dates:
            lvls[pres][var] = np.append(lvls[pres][var],di2[date][var][q])

for e,var in enumerate(np.append(varS,'THEstar')):
    for q,pres in enumerate(np.arange(100,1001,25)):
        av[var] = np.append(av[var],np.mean(lvls[pres][var]))
    ax[e].plot(av[var],np.arange(100,1001,25),color='k',linewidth=2)
    ax[e].set_xlim(xlims[e])
    ax[e].set_xlabel(xlab[e])

ax[0].axvline(0,color='grey',linestyle='-')
ax[0].set_xticks(ticks=np.arange(-1,11))
ax[0].set_xticklabels(labels=np.array(['',0,'',2,'',4,'',6,'',8,'',10]))
ax[0].set_ylim(100,1000)
ax[0].set_ylabel('pressure [hPa]')

ax[0].invert_yaxis()
path = '/home/ascherrmann/010-IFS/'
fig.savefig(path + 'vertical-averages-black.png',dpi=300,bbox_inches="tight")
plt.close()

