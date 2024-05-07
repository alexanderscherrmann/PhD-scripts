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

fig1,ax = plt.subplots(figsize=(8,6))
fig2,ax2 = plt.subplots(figsize=(8,6))
fig3,ax3 = plt.subplots(figsize=(8,6))
fig4,ax4 = plt.subplots(figsize=(8,6))
fig5,ax5 = plt.subplots(figsize=(8,6))
legend=np.array(['DJF','MAM','JJA','SON'])
colors=['b','darkgreen','r','saddlebrown','dodgerblue','dodgerblue','royalblue']
col200km = ['deepskyblue','limegreen','lightcoral','peru']
ENSW = np.array(['east-','north','south','west-'])
enswmarker = ['>','^','v','<']

markers=['o','o','o','o','+','x','d']
ls=' '
for co,ma in zip(colors,markers):
    ax.plot([],[],ls=ls,marker=ma,color=co)
    ax2.plot([],[],ls=ls,marker=ma,color=co)
    ax3.plot([],[],ls=ls,marker=ma,color=co)
    ax4.plot([],[],ls=ls,marker=ma,color=co)
    ax5.plot([],[],ls=ls,marker=ma,color=co)

for co,ma in zip(col200km,markers):
    ax.plot([],[],ls=ls,marker=ma,color=co)
    ax2.plot([],[],ls=ls,marker=ma,color=co)
    ax3.plot([],[],ls=ls,marker=ma,color=co)
    ax4.plot([],[],ls=ls,marker=ma,color=co)
    ax5.plot([],[],ls=ls,marker=ma,color=co)

errorbars = dict()
base = dict()
amps = [0.7, 1.4, 2.1, 2.8, 3.5, 4.2,0.9,1.7,1.1,0.3,0.5]
dist = np.array([200, 400])
var = ['slp','t-genesis','pvampl','precip']

for sea in legend:
    errorbars[sea] = dict()
    base[sea] = dict()
    for amp in amps:
        base[sea][amp] = dict()
        for v in var:
            base[sea][amp][v] = dict()

    for dis in dist:
        errorbars[sea][dis] = dict()

        for amp in amps:
            errorbars[sea][dis][amp] = dict()
            
            for v in var:
                errorbars[sea][dis][amp][v] = dict()
                errorbars[sea][dis][amp][v]['med'] = np.array([])
                errorbars[sea][dis][amp][v]['at'] = np.array([])

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
    if sim[4:8]=='clim':
        ax3.scatter(aminslp,genesis,color=col,marker=mark)

    if sim[4:7]=='200':
       qq = np.where(legend==sea)[0][0]
       col = col200km[qq]

    if np.any(ENSW==sim[11:16]):
        qq = np.where(ENSW==sim[11:16])[0][0]
        mark = enswmarker[qq]

    if sim[4:8]=='clim':
        ax2.scatter(aminslp,slpmin,color=col,marker=mark,s=amp*10-3)
        ax.scatter(maxpv,slpmin,color=col,marker=mark,s=amp*10-3)
#    if sim[4:8]=='clim':

#        ax2.text(aminslp,slpmin,'%.1f'%(amp))
    
        ax5.scatter(atmt,medmt,color=col,marker=mark)
        ax5.text(atmt,medmt,'%.1f'%(amp))

    if sim[4:8]=='clim':
        base[sea][amp]['slp']['med'] = slpmin
        base[sea][amp]['slp']['at'] = aminslp
        base[sea][amp]['pvampl']['med'] = maxpv
        base[sea][amp]['precip']['med'] = prec[sim]['totalprecip']-prec[sea+'-clim']['totalprecip']


    else:
        errorbars[sea][dis][amp]['slp']['med'] = np.append(errorbars[sea][dis][amp]['slp']['med'],slpmin)
        errorbars[sea][dis][amp]['slp']['at'] = np.append(errorbars[sea][dis][amp]['slp']['at'],aminslp)
        errorbars[sea][dis][amp]['pvampl']['med'] = np.append(errorbars[sea][dis][amp]['pvampl']['med'],maxpv)
        errorbars[sea][dis][amp]['precip']['med'] = np.append(errorbars[sea][dis][amp]['precip']['med'],prec[sim]['totalprecip']-prec[sea+'-clim']['totalprecip'])

for col,sea,q in zip(colors,legend,range(len(colors))):
 for dis in dist[::-1]:
  for amp in amps[:3]:
    if sea=='JJA' and dis==200 and amp==0.7:
        continue
    yerr = np.array([[base[sea][amp]['slp']['med']-np.min(errorbars[sea][dis][amp]['slp']['med'])],[np.max(errorbars[sea][dis][amp]['slp']['med'])-base[sea][amp]['slp']['med']]])
    xerr = np.array([[base[sea][amp]['slp']['at']-np.min(errorbars[sea][dis][amp]['slp']['at'])],[np.max(errorbars[sea][dis][amp]['slp']['at'])-base[sea][amp]['slp']['at']]])

    if np.any(yerr<0):
        yerr[yerr<0]=0
    if np.any(xerr<0):
        xerr[xerr<0]=0

    co = col
    if int(dis)==200:
        col = col200km[q]

    ax2.errorbar(base[sea][amp]['slp']['at'],base[sea][amp]['slp']['med'],yerr=yerr,xerr=xerr,ecolor=col)

    xerr = np.array([[base[sea][amp]['pvampl']['med']-np.min(errorbars[sea][dis][amp]['pvampl']['med'])],[np.max(errorbars[sea][dis][amp]['pvampl']['med'])-base[sea][amp]['pvampl']['med']]])
    if np.any(xerr<0):
        xerr[xerr<0]=0

    ax.errorbar(base[sea][amp]['pvampl']['med'],base[sea][amp]['slp']['med'],yerr=yerr,xerr=xerr,ecolor=col)

    yerr = np.array([[base[sea][amp]['precip']['med']-np.min(errorbars[sea][dis][amp]['precip']['med'])],[np.max(errorbars[sea][dis][amp]['precip']['med'])-base[sea][amp]['precip']['med']]])
    xerr = np.array([[base[sea][amp]['pvampl']['med']-np.min(errorbars[sea][dis][amp]['pvampl']['med'])],[np.max(errorbars[sea][dis][amp]['pvampl']['med'])-base[sea][amp]['pvampl']['med']]])

    ax4.errorbar(base[sea][amp]['pvampl']['med'],base[sea][amp]['precip']['med'],yerr=yerr,xerr=xerr,ecolor=col)
for col,sea,q in zip(colors,legend,range(len(colors))):
 for dis in dist[:1]:
  for amp in amps[:3]:
    ax2.scatter(base[sea][amp]['slp']['at'],base[sea][amp]['slp']['med'],marker='o',s=amp*10-3,color=col,zorder=10)
    ax.scatter(base[sea][amp]['pvampl']['med'],base[sea][amp]['slp']['med'],marker='o',s=amp*10-3,color=col,zorder=10)
    ax4.scatter(base[sea][amp]['pvampl']['med'],base[sea][amp]['precip']['med'],marker='o',s=amp*10-3,color=col,zorder=10)

ax.set_xlabel('PV anomaly [PVU]')
ax.set_ylabel('MED cyclone min SLP [hPa]')
ax.legend(legend,loc='upper right')
name = dwrf + 'image-output/errorbar2-MED-cyclone-PV-anomaly-SLP-scatter-min-of-all-tracks-in-MED.png'
fig1.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig1)

ax2.set_xlabel('Atlantic cyclone min SLP [hPa]')
ax2.set_ylabel('MED cyclone min SLP [hPa]')
ax2.legend(legend,loc='upper left')
name = dwrf + 'image-output/errorbar2-Atlantic-MED-cyclone-SLP-scatter-min-of-all-tracks-in-MED.png'
fig2.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig2)

#ax3.set_xlabel('MED cyclone genesis lon [$^{\circ}$]')
#ax3.set_ylabel('PV anomaly [PVU]')
ax3.set_xlabel('Atlantic cyclone min SLP [hPa]')
ax3.set_ylabel('Med cyclone genesis time [h]')
ax3.legend(legend,loc='upper right')
name = dwrf + 'image-output/errorbar2-MED-cyclone-genesis-time-scatter-min-of-all-tracks-in-MED.png'
#fig3.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig3)

ax4.set_xlabel('PV anomaly [PVU]')
ax4.set_ylabel('precipitation [m]')
ax4.legend(legend,loc='upper left')
name = dwrf + 'image-output/errorbar2-precipitation-vs-PV-anomaly.png'
fig4.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig4)


ax5.set_xlabel('time of mature stage of Atlantic cyclone [h]')
ax5.set_ylabel('time of mature stage of Mediterranean cyclone [h]')
ax5.legend(legend,loc='upper right')
name = dwrf + 'image-output/seasons-MED-cyclone-mature-time-min-of-all-tracks-in-MED.png'
#fig5.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig5)

