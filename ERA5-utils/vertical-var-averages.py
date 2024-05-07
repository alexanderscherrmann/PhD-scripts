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
p = '/home/ascherrmann/009-ERA-5/'

path=p
dates = np.array([])
dat = np.loadtxt(p + 'manos-test-data.txt')
IDS = dat[:,-2].astype(int)

LON=np.linspace(-180,180,721)
LAT=np.linspace(-90,90,361)

p = '/home/ascherrmann/009-ERA-5/'

f = open(p + 'lons-Manos.txt',"rb")
lons = pickle.load(f)
f.close()

f = open(p + 'lats-Manos.txt',"rb")
lats = pickle.load(f)
f.close()

f = open(p + 'dates-Manos.txt',"rb")
Dates = pickle.load(f)
f.close()

f = open(p + 'SLPs-Manos.txt',"rb")
SLPs = pickle.load(f)
f.close()

dates = np.array([])
Lon = np.array([])
Lat = np.array([])
SLP = np.array([])
for k in range(len(Dates)):
    dates = np.append(dates,Dates[k][-1])

    Lon = np.append(Lon,lons[k][-1])
    Lat = np.append(Lat,lats[k][-1])
    SLP = np.append(SLP,SLPs[k][-1])

rdis=200

varS = np.array(['PV','TH','THE'])
varP = np.array(['T'])
di = dict()
di2= dict()
tmpd = dict()

for ul, date in enumerate(dates):
    ID = IDS[ul]
    di[date] = dict()
    di2[date] = dict()

    ID = helper.MED_cyclones_date_to_id(date)
    yyyy = int(date[0:4])
    MM = int(date[4:6])
    DD = int(date[6:8])
    hh = int(date[9:])

    ana_path='/home/ascherrmann/009-ERA-5/Manos-test/'
    clat2 = np.where(LAT==Lat[ul])[0].astype(int)
    clon2 = np.where(LON==Lon[ul])[0].astype(int)

    clat = clat2 + helper.radial_ids_around_center_calc_ERA5(rdis)[1]
    clon = clon2 + helper.radial_ids_around_center_calc_ERA5(rdis)[0]

    sfile = ana_path + 'S' + date
    pfile = ana_path + 'P' + date

    s = xr.open_dataset(sfile, drop_variables=['P','RH','VORT','PVRCONVT','PVRCONVM','PVRTURBT','PVRTURBM','PVRLS','PVRCOND','PVRSW','PVRLWH','PVRLWC','PVRDEP','PVREVC','PVREVR','PVRSUBI','PVRSUBS','PVRMELTI','PVRMELTS','PVRFRZ','PVRRIME','PVRBF'])
    p = xr.open_dataset(pfile,drop_variables=['SWC', 'RWC', 'IWC', 'LWC','PS','CC','OMEGA','tsw','tlw','tmix','tconv','tcond','tdep','tbf','tevc','tsubi','tevr','tsubs','tmelti','tmelts','tfrz','trime','udotconv','vdotconv','udotmix','vdotmix','tls','tce'])
    PS = s.PS.values[0,clat,clon]
    for q in varS:
        tmp = getattr(s,q)
        di[date][q] = np.transpose(tmp.values[0,:,clat,clon])
        di2[date][q] = np.array([])

    for q in varP:
        tmp = getattr(p,q)
        di[date][q] = np.transpose(tmp.values[0,:,clat,clon])
        di2[date][q] = np.array([])
    hya = s.hyai.values
    hyb = s.hybi.values


    di2[date]['THEstar'] = np.array([])
    for pres in np.arange(100,1001,25):
        for q in np.append(varS,varP):
            tmpd[q] = np.array([])
        for e in range(len(clat)):
            P = helper.modellevel_ERA5(PS[e],hya,hyb)
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
#    ax[0].plot(di2[date]['PV'],np.arange(100,1001,25),color='grey')
    for w,q in enumerate(pltvar[:]):
        ax[w].plot(di2[date][q],np.arange(100,1001,25),color=cm.bwr((SLP[k]-np.min(SLP))/(np.max(SLP)-np.min(SLP))),linewidth=2)
#        if date=='20181011_22':
#            ax[w].plot(di2[date][q],np.arange(100,1001,25),color='red')
#        if date=='20181204_20':
#            ax[w].plot(di2[date][q],np.arange(100,1001,25),color='blue')
#        if date=='20171214_02':
#            ax[w].plot(di2[date][q],np.arange(100,1001,25),color='green')


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

fig.savefig(path + 'vertical-averages.png',dpi=300,bbox_inches="tight")
plt.close()

