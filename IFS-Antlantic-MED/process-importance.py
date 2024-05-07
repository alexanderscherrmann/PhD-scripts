import numpy as np
import pickle

import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import helper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patch
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import cm


CT = 'ETA'
rdis = 800
pload ='/home/ascherrmann/010-IFS/ctraj/' + CT +'/use/'

f = open(pload + 'PV-data-' + CT + 'dPSP-100-ZB-800PVedge-0.3-' + str(rdis) + '-correct-distance.txt','rb')
data = pickle.load(f)
f.close()

datadi = data['rawdata']
dipv = data['dipv']
dit = data['dit']
labs = helper.traced_vars_IFS()

proc = ['APVTOT','PVRCONVT','PVRTURBT','PVRCONVM','PVRTURBM','PVRLS','APVRAD']#'PVRLWH','PVRLWC']
lab = [r'$\Delta$PV$_0$',r'APV$_{\mathrm{TOT}}$','RES',r'APV$_{\mathrm{CONVT}}$',r'APV$_{\mathrm{TURBT}}$',r'APV$_{\mathrm{CONVM}}$',r'APV$_{\mathrm{TURBM}}$',r'APV$_{\mathrm{LS}}$',r'APV$_{\mathrm{RAD}}$']#LWH','LWC']
### use CONVT in first and TURBT in second PVR-T
trajbox = []
meanbox = []
meanboxcyc = []
meanboxenv = []

trajdi = dict()
meandi = dict()
meanc = dict()
meane = dict()

meanres = np.array([])
meancres = np.array([])
meaneres = np.array([])
meanpv = np.array([])
meanpvc = np.array([])
meanpve = np.array([])

for pr in proc:
    trajdi[pr] = np.array([])
    meandi[pr] = np.array([])
    meanc[pr] = np.array([])
    meane[pr] = np.array([])

for ul, date in enumerate(datadi.keys()):
    idp = np.where(datadi[date]['PV'][:,0]>=0.75)[0]
    for pr in proc:
        trajdi[pr] = np.append(trajdi[pr],dipv[date]['cyc'][pr][idp,0] + dipv[date]['env'][pr][idp,0])
        meandi[pr] = np.append(meandi[pr],np.mean(dipv[date]['cyc'][pr][idp,0]+ dipv[date]['env'][pr][idp,0]))
        meanc[pr] = np.append(meanc[pr],np.mean(dipv[date]['cyc'][pr][idp,0]))
        meane[pr] = np.append(meane[pr],np.mean(dipv[date]['env'][pr][idp,0]))

    datadi[date]['RES'] = np.zeros(datadi[date]['PV'].shape)
    datadi[date]['PVRTOT'] = np.zeros(datadi[date]['PV'].shape)
    for pv in labs[8:]:
        datadi[date]['PVRTOT']+=datadi[date][pv]
    RES = np.flip(np.cumsum(np.flip(datadi[date]['PVRTOT'][:,1:],axis=1),axis=1),axis=1)-dipv[date]['env']['deltaPV'][:,:-1] - dipv[date]['cyc']['deltaPV'][:,:-1]#np.flip(np.cumsum(np.flip(datadi[date]['DELTAPV'][:,1:],axis=1),axis=1),axis=1)
    CRES = np.flip(np.cumsum(np.flip(datadi[date]['PVRTOT'][:,1:]* dit[date]['cyc'][:,1:],axis=1),axis=1),axis=1)-dipv[date]['env']['deltaPV'][:,:-1]#np.flip(np.cumsum(np.flip(datadi[date]['DELTAPV'][:,1:]* dit[date]['cyc'][:,1:],axis=1),axis=1),axis=1)
    ERES = np.flip(np.cumsum(np.flip(datadi[date]['PVRTOT'][:,1:] * dit[date]['env'][:,1:],axis=1),axis=1),axis=1)-dipv[date]['env']['deltaPV'][:,:-1]#np.flip(np.cumsum(np.flip(datadi[date]['DELTAPV'][:,1:]* dit[date]['env'][:,1:],axis=1),axis=1),axis=1)
    meanres = np.append(meanres,np.mean(RES[idp,0]))
    meancres = np.append(meancres,np.mean(CRES[idp,0]))
    meaneres = np.append(meaneres,np.mean(ERES[idp,0]))
    meanpv = np.append(meanpv,np.mean(dipv[date]['cyc']['deltaPV'][idp,0] + dipv[date]['env']['deltaPV'][idp,0]))#datadi[date]['PV'][idp,0]))
    meanpvc = np.append(meanpvc,np.mean(dipv[date]['cyc']['deltaPV'][idp,0]))
    meanpve=  np.append(meanpve,np.mean(dipv[date]['env']['deltaPV'][idp,0]))


meanbox.append(np.sort(meanpv))
meanboxcyc.append(np.sort(meanpvc))
meanboxenv.append(np.sort(meanpve))


###
###

for pr in proc[:1]:
    trajbox.append(np.sort(trajdi[pr]))
    meanbox.append(np.sort(meandi[pr]))
    meanboxcyc.append(np.sort(meanc[pr]))
    meanboxenv.append(np.sort(meane[pr]))

meanbox.append(np.sort(meanres))
meanboxcyc.append(np.sort(meancres))
meanboxenv.append(np.sort(meaneres))

###
###
for pr in proc[1:]:
    trajbox.append(np.sort(trajdi[pr]))
    meanbox.append(np.sort(meandi[pr]))
    meanboxcyc.append(np.sort(meanc[pr]))
    meanboxenv.append(np.sort(meane[pr]))


flier = dict(marker='+',markerfacecolor='grey',markersize=1,linestyle=' ',markeredgecolor='grey')
meanline = dict(linestyle='-',linewidth=1,color='red')
meanline2 = dict(linestyle='-',linewidth=1,color='dodgerblue')
capprops = dict(linestyle='-',linewidth=1,color='grey')
medianprops = dict(linestyle='-',linewidth=1,color='black')
boxprops = dict(linestyle='-',linewidth=1.,color='slategrey')
whiskerprops= dict(linestyle='-',linewidth=1,color='grey')

#fig, ax = plt.subplots()
#bp = ax.boxplot(trajbox,labels=lab,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops)
#ax.set_ylabel(r'PV [PVU]')
#ax.set_xlim(0,8)
#ax.set_ylim(-2,2)
#ax.set_xticklabels(labels=lab)
#ax.xticks(rotation=90)
#fig.savefig('/home/ascherrmann/010-IFS/' + 'dominant-process-traj.png',dpi=300,bbox_inches="tight")
#plt.close('all')

fig, ax = plt.subplots()
bp = ax.boxplot(meanbox,whis=(10,90),labels=lab,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops)
ax.axhline(0,color='grey',zorder=0)
ax.set_ylabel(r'PV [PVU]')
ax.set_xlim(0,10)
ax.set_ylim(-2,2)
ax.set_xticklabels(labels=lab)
plt.xticks(rotation=90)
ax.text(0.03, 0.95, 'a)', transform=ax.transAxes,fontsize=12, fontweight='bold',va='top')
fig.savefig('/home/ascherrmann/010-IFS/' + CT + '-dominant-mean-traj.png',dpi=300,bbox_inches="tight")
plt.close('all')

fig, axes = plt.subplots(2,1,sharex=True)
axes = axes.flatten()
ax = axes[0]
ax.set_ylim(-1.5,1.5)

ax.text(0.03, 0.95, 'b)', transform=ax.transAxes,fontsize=12, fontweight='bold',va='top')
bp = ax.boxplot(meanboxcyc,whis=(10,90),labels=lab,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops)
ax.axhline(0,color='grey',zorder=0)
ax.set_ylabel(r'cyc. PV [PVU]')
ax.set_xlim(0,10)
#ax.set_xticks(ticks=np.arange(1,9))
#ax.set_xticklabels(labels=lab)
#
ax = axes[1]
ax.text(0.03, 0.95, 'c)', transform=ax.transAxes,fontsize=12, fontweight='bold',va='top')
bp = ax.boxplot(meanboxenv,whis=(10,90),labels=lab,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops)
ax.set_ylabel(r'env. PV [PVU]')
ax.axhline(0,color='grey',zorder=0)
ax.set_xlim(0,10)
ax.set_ylim(-1.5,1.5)
plt.xticks(rotation=90)
ax.set_xticklabels(labels=lab)

plt.subplots_adjust(left=0.1,hspace=0.0,wspace=0)
fig.savefig('/home/ascherrmann/010-IFS/' + CT + '-dominant-mean-cyc-env.png',dpi=300,bbox_inches="tight")
plt.close('all')

