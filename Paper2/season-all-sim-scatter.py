import wrf
from netCDF4 import Dataset as ds
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import wrfsims
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle

SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)

f = open('/atmosdyn2/ascherrmann/013-WRF-sim/precipitation-dict.txt','rb')
prec = pickle.load(f)
f.close()

dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/home/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'
image = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'
ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]

### time of slp measure of atlantic cyclone
hac = 12

PVpos = [[93,55],[95,56],[140,73],[125,72]]
steps = [[0,0],[4,0],[-4,0],[0,4],[0,-4],[8,0],[-8,0],[0,8],[0,-8]]

names = np.array([])
MINslp = np.array([])
MAXpv = np.array([])

fig1,axes = plt.subplots(2,2,figsize=(8,6),sharex=True,sharey=True)
axes = axes.flatten()
fig2,axes2 = plt.subplots(2,2,figsize=(8,6),sharex=True,sharey=True)
axes2 = axes2.flatten()
fig3,axes3 = plt.subplots(2,2,figsize=(8,6),sharex=True,sharey=True)
axes3 = axes3.flatten()

legend=np.array(['DJF','MAM','JJA','SON'])
colors=['b','darkgreen','r','saddlebrown','dodgerblue','dodgerblue','royalblue']
col200km = ['deepskyblue','limegreen','lightcoral','peru']
ENSW = np.array(['east-','north','south','west-'])
enswmarker = ['1','d','s','x']

markers=['o','o','o','o','+','x','d']
ls=' '
for axe in [axes,axes2,axes3]:
 ax=axe[0]
 for co,ma in zip(colors,markers):
    ax.plot([],[],ls=ls,marker=ma,color=co)

amps = [0.7, 1.4, 2.1, 2.8, 3.5, 4.2,0.9,1.7,1.1,0.3,0.5]
dist = np.array([200, 400])
var = ['slp','t-genesis','pvampl','precip']

for simid, sim in enumerate(SIMS):

    if "-not" in sim:
        continue
    if sim[-4:]=='clim':
        continue
    
    strings=np.array([])
    for l in re.findall(r"[-+]?(?:\d*\.\d+|\d+)",sim):
        strings = np.append(strings,float(l[1:]))

    amp = float(strings[-1])
    sea = sim[:3]
    if np.any(dist==int(strings[0])):
        dis = int(strings[0])

    medid = MEDIDS[simid]
    atid = ATIDS[simid]

    if len(medid)==0 or len(atid)==0:
        continue
    
    whichax=np.where(legend==sea)[0][0]

    print(sim)
    ic = ds(dwrf + sim + '/wrfout_d01_2000-12-01_00:00:00')
    tra = np.loadtxt(tracks + sim + '-filter.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    deepening = tra[:,-2]
    IDs = tra[:,-1]
    minslp=np.array([])
    isd = np.array([])
    genesis = np.array([]) 
    medmt = np.array([])
    for mei in medid:
        loc = np.where((IDs==mei) & (t<204))[0]
        minslp = np.append(minslp,np.min(slp[loc]))
        genesis = np.append(genesis,np.min(t[loc]))
        medmt = np.append(medmt,t[loc[np.argmin(slp[loc])]])

    slpmin = np.min(minslp)
    genesis = np.min(genesis)
    medmt = medmt[np.argmin(minslp)]

    PV = wrf.getvar(ic,'pvo')
    p = wrf.getvar(ic,'pressure')

    pv = wrf.interplevel(PV,p,300,meta=False)
    maxpv = np.zeros(len(PVpos)*len(steps))
    for q,l in enumerate(PVpos):
        for qq,st in enumerate(steps):
            maxpv[q*len(PVpos)+qq] = pv[l[1]+st[1],l[0]+st[0]]

    maxpv = np.max(maxpv)
    mark='o'
    col = 'k'
    if sim.startswith('DJF'):
        col = 'b'
    if sim.startswith('MAM'):
        col = 'seagreen'
    if sim.startswith('JJA'):
        col = 'r'
    if sim.startswith('SON'):
        col = 'saddlebrown'

    
    aslp = np.array([])
    atmt = np.array([])
    for ai in atid:
        loc = np.where(IDs==ai)[0]
        aslp = np.append(aslp,np.min(slp[loc]))
        atmt = np.append(atmt,t[loc[np.argmin(slp[loc])]])
    
    aminslp = np.min(aslp)

    atmt = atmt[np.argmin(aslp)]

    if sim[4:7]=='200':
       qq = np.where(legend==sea)[0][0]
       col = col200km[qq]

    if np.any(ENSW==sim[11:16]):
        qq = np.where(ENSW==sim[11:16])[0][0]
        mark = enswmarker[qq]

#    if sim[4:8]=='clim':
    axes2[whichax].scatter(aminslp,slpmin,color=col,marker=mark,s=amp*10-3)
    axes[whichax].scatter(maxpv,slpmin,color=col,marker=mark,s=amp*10-3)
#    axes3[whichax].scatter(atmt,medmt,color=col,marker=mark)
#        ax5.text(atmt,medmt,'%.1f'%(amp))

#ax.set_xlabel('PV anomaly [PVU]')
#ax.set_ylabel('MED cyclone min SLP [hPa]')
axes[0].legend(legend,loc='upper right')
axes[0].set_ylabel('MED cyclone min SLP [hPa]')
axes[2].set_ylabel('MED cyclone min SLP [hPa]')
axes[2].set_xlabel('PV anomaly [PVU]')
axes[3].set_xlabel('PV anomaly [PVU]')
fig1.subplots_adjust(wspace=0,hspace=0)
name = image + 'errorbar2-MED-cyclone-PV-anomaly-SLP-scatter-min-of-all-tracks-in-MED.png'
fig1.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig1)

#ax2.set_xlabel('Atlantic cyclone min SLP [hPa]')
#ax2.set_ylabel('MED cyclone min SLP [hPa]')
axes2[0].legend(legend,loc='upper left')
axes2[0].set_ylabel('MED cyclone min SLP [hPa]')
axes2[2].set_ylabel('MED cyclone min SLP [hPa]')
axes2[2].set_xlabel('Atlantic cyclone min SLP [hPa]')
axes2[3].set_xlabel('Atlantic cyclone min SLP [hPa]')
fig2.subplots_adjust(wspace=0,hspace=0)
name = image + 'errorbar2-Atlantic-MED-cyclone-SLP-scatter-min-of-all-tracks-in-MED.png'
fig2.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig2)

#ax3.set_xlabel('PV anomaly [PVU]')
#ax3.set_ylabel('precipitation [m]')
#axes3[0].legend(legend,loc='upper left')
#axes3[0].set_ylabel('PV anomaly [PVU]')
#axes3[2].set_ylabel('PV anomaly [PVU]')
#axes3[2].set_xlabel('precipitation [m]')
#axes3[3].set_xlabel('precipitation [m]')
#fig3.subplots_adjust(wspace=0,hspace=0)
#name = image + 'errorbar2-precipitation-vs-PV-anomaly.png'
#fig3.savefig(name,dpi=300,bbox_inches='tight')
#plt.close(fig3)


