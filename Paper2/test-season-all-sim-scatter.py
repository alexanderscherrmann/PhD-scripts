import wrf
from netCDF4 import Dataset as ds
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import wrfsims
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
from scipy.stats import pearsonr as pr

SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)

f = open('/atmosdyn2/ascherrmann/013-WRF-sim/precipitation-dict.txt','rb')
prec = pickle.load(f)
f.close()

dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'
image = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'
ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]

def find_nearest_grid_point(lon,lat):

    dlon = LON-lon
    dlat = LAT-lat

    lonid = np.where(abs(dlon)==np.min(abs(dlon)))[0][0]
    latid = np.where(abs(dlat)==np.min(abs(dlat)))[0][0]

    return lonid,latid


MSLP = dict()
for sea in ['DJF','MAM','JJA','SON']:
    ref = ds(dwrf + '%s-clim/wrfout_d01_2000-12-01_00:00:00'%sea)
    MSLP[sea] = ref.variables['MSLP'][0,:]

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
PVarr0 = dict()
PVarr12 = dict()
PVarr24 = dict()
PVstream0 = dict()
PVstream6 = dict()
PVstream12 = dict()
PVmat = dict()
deepen0 = dict()
deepen12 = dict()
deepen24 = dict()
minSLP = dict()
for sea in legend:
    PVarr0[sea] =np.array([])
    PVarr24[sea] =np.array([])
    PVarr12[sea] =np.array([])
    PVstream0[sea] =np.array([])
    PVstream6[sea] =np.array([])
    PVstream12[sea] =np.array([])
    PVmat[sea] = np.array([])

    deepen0[sea] = np.array([])
    deepen24[sea] = np.array([])
    deepen12[sea] = np.array([])
    minSLP[sea] = np.array([])

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
    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]
    minslp=np.array([])
    isd = np.array([])
    genesis = np.array([]) 
    medmt = np.array([])
    loc = np.where(IDs==2)[0]
    mlon,mlat = tlon[loc[np.argmin(slp[loc])]],tlat[loc[np.argmin(slp[loc])]]
    slpmin = np.min(slp[loc])

    minSLP[sea] = np.append(minSLP[sea],slpmin)
    lonid,latid = find_nearest_grid_point(mlon,mlat)
    slpmin -=MSLP[sea][latid,lonid]


    PV = wrf.getvar(ic,'pvo')
    p = wrf.getvar(ic,'pressure')

    pv = wrf.interplevel(PV,p,300,meta=False)
    maxpv = np.zeros(len(PVpos)*len(steps))
    avinPV = np.zeros(len(PVpos)*len(steps))
    for q,l in enumerate(PVpos):
        for qq,st in enumerate(steps):
            maxpv[q*len(PVpos)+qq] = pv[l[1]+st[1],l[0]+st[0]]
            avinPV[q*len(PVpos)+qq] = np.mean(pv[l[1]-4:l[1]+5,l[0]-4:l[0]+5])


    averageinitialPV = np.max(avinPV)
    maxpv = np.max(maxpv)
    mark='o'

    dm0 = helper.simulation_time_to_day_string(t[loc][0])
    dm6 = helper.simulation_time_to_day_string(t[loc][2])
    dm12 = helper.simulation_time_to_day_string(t[loc][4])
    dmmat = helper.simulation_time_to_day_string(t[loc][np.argmin(slp[loc])])

    gen0h = ds(dwrf+sim+'/wrfout_d01_2000-12-%s:00:00'%dm0)
    gen6h = ds(dwrf+sim+'/wrfout_d01_2000-12-%s:00:00'%dm6)
    gen12h = ds(dwrf+sim+'/wrfout_d01_2000-12-%s:00:00'%dm12)
    mature = ds(dwrf+sim+'/wrfout_d01_2000-12-%s:00:00'%dmmat)
    
    PV = wrf.getvar(gen0h,'pvo')
    pres = wrf.getvar(gen0h,'pressure')
    PV300 = wrf.interplevel(PV,pres,300,meta=False)
    lo,la = find_nearest_grid_point(tlon[loc][0],tlat[loc][0])
    avstreamerPV0 = np.mean(np.sort(PV300[la-4:la+5,lo-4:lo+5].flatten())[int((np.sort(PV300[la-4:la+5,lo-4:lo+5]).flatten()).size*0.8)])

    PV = wrf.getvar(gen6h,'pvo')
    pres = wrf.getvar(gen6h,'pressure')
    PV300 = wrf.interplevel(PV,pres,300,meta=False)

    lo,la = find_nearest_grid_point(tlon[loc][2],tlat[loc][2])
    avstreamerPV6 = np.mean(np.sort(PV300[la-4:la+5,lo-4:lo+5].flatten())[int((np.sort(PV300[la-4:la+5,lo-4:lo+5]).flatten()).size*0.8)])

    PV = wrf.getvar(gen12h,'pvo')
    pres = wrf.getvar(gen12h,'pressure')
    PV300 = wrf.interplevel(PV,pres,300,meta=False)

    lo,la = find_nearest_grid_point(tlon[loc][4],tlat[loc][4])
    avstreamerPV12 = np.mean(np.sort(PV300[la-4:la+5,lo-4:lo+5].flatten())[int((np.sort(PV300[la-4:la+5,lo-4:lo+5]).flatten()).size*0.8)])

    PV = wrf.getvar(mature,'pvo')
    pres = wrf.getvar(mature,'pressure')
    PV300 = wrf.interplevel(PV,pres,300,meta=False)

    lo,la = find_nearest_grid_point(tlon[loc][np.argmin(slp[loc])],tlat[loc][np.argmin(slp[loc])]) 
    avstreamerPVmature = np.mean(np.sort(PV300[la-4:la+5,lo-4:lo+5].flatten())[int((np.sort(PV300[la-4:la+5,lo-4:lo+5]).flatten()).size*0.8)])
    
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
    loc = np.where(IDs==1)[0]
    aslp = np.append(aslp,np.min(slp[loc]))
    atmt = np.append(atmt,t[loc[np.argmin(slp[loc])]])
    

    aminslp = np.min(aslp)

    atmt = atmt[np.argmin(aslp)]
    
    loc = np.where(IDs==2)[0]
    dSLP0 = slp[loc][8]-slp[loc][0]

    if sim[4:7]=='200':
       qq = np.where(legend==sea)[0][0]
       col = col200km[qq]

    if np.any(ENSW==sim[11:16]):
        qq = np.where(ENSW==sim[11:16])[0][0]
        mark = enswmarker[qq]

#    if sim[4:8]=='clim':
    axes2[whichax].scatter(aminslp,slpmin,color=col,marker=mark,s=amp*10-3)
    axes[whichax].scatter(averageinitialPV,slpmin,color=col,marker=mark,s=amp*10-3)
    axes3[whichax].scatter(averageinitialPV,dSLP0,color=col,marker=mark)
    PVarr0[sea] = np.append(PVarr0[sea],averageinitialPV)
    deepen0[sea] = np.append(deepen0[sea],dSLP0)
    PVstream0[sea] =np.append(PVstream0[sea],avstreamerPV0)
    PVstream6[sea] =np.append(PVstream6[sea],avstreamerPV6)
    PVstream12[sea] =np.append(PVstream12[sea],avstreamerPV12)
    PVmat[sea] = np.append(PVmat[sea],avstreamerPVmature)

    try:
        dSLP12 = slp[loc][12]-slp[loc][4]
    except:
        continue
    PVarr12[sea] = np.append(PVarr12[sea],averageinitialPV)
    deepen12[sea] = np.append(deepen12[sea],dSLP12)
    try:
        dSLP24 = slp[loc][16]-slp[loc][8]
    except:
        continue
    PVarr24[sea] = np.append(PVarr24[sea],averageinitialPV)
    deepen24[sea] = np.append(deepen24[sea],dSLP24)

#        ax5.text(atmt,medmt,'%.1f'%(amp))

#ax.set_xlabel('PV anomaly [PVU]')
#ax.set_ylabel('MED cyclone min SLP [hPa]')
axes[0].legend(legend,loc='upper right')
axes[0].set_ylabel('MED cyclone min SLP [hPa]')
axes[2].set_ylabel('MED cyclone min SLP [hPa]')
axes[2].set_xlabel('av PV @jet [PVU]')
axes[3].set_xlabel('av PV @jet [PVU]')
fig1.subplots_adjust(wspace=0,hspace=0)
name = image + 'test-errorbar-MED-cyclone-PV-anomaly-SLP-scatter-min-of-all-tracks-in-MED.png'
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
name = image + 'test-errorbar-Atlantic-MED-cyclone-SLP-scatter-min-of-all-tracks-in-MED.png'
fig2.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig2)

for sea in legend:
    print(sea,pr(PVarr0[sea],deepen0[sea]))
    print(pr(PVarr12[sea],deepen12[sea]))
    print(pr(PVarr24[sea],deepen24[sea]))

for sea in legend:
    print(sea,pr(PVstream0[sea],deepen0[sea]))
    print(pr(PVstream0[sea],minSLP[sea]))


#ax3.set_xlabel('PV anomaly [PVU]')
#ax3.set_ylabel('precipitation [m]')
axes3[0].legend(legend,loc='upper left')
axes3[0].set_ylabel('deepening rate [hPa (24 h)$^{-1}$]')
axes3[2].set_ylabel('deepening rate [hPa (24 h)$^{-1}$]')
axes3[2].set_xlabel('av PV @jet [PVU]')
axes3[3].set_xlabel('av PV @jet [PVU]')
fig3.subplots_adjust(wspace=0,hspace=0)
name = image + 'MED-cyclone-deepening-vs-PV-anomaly.png'
fig3.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig3)


