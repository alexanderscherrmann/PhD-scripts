import numpy as np
import pickle
import os
import matplotlib
from matplotlib import pyplot as plt
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper

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
import matplotlib.gridspec as gridspec
import cartopy

CT = 'ETA'

pload = '/home/ascherrmann/010-IFS/ctraj/' + CT + '/use/'

rdis = 800
f = open(pload + 'PV-data-'+CT + 'dPSP-100-ZB-800PVedge-0.3-%d-correct-distance.txt'%rdis,'rb')
PVdata = pickle.load(f)
f.close()

dipv = PVdata['dipv']
dit = PVdata['dit']

adv = np.array([])
cyc = np.array([])
env = np.array([])
c = 'cyc'
e = 'env'

f = open('/home/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','rb')
locdata =  pickle.load(f)
f.close()

MON = np.array([])
for d in os.listdir(pload[:-4]):
    if(d.startswith('traend-')):
        MON = np.append(MON,d)
MON = np.sort(MON)
PVstart = np.array([])
PVend = np.array([])
ca = 0
ct = 0
ol = np.array([])
ol2 = np.array([])
pvloc = dict()
ac = dict()
pressuredi = dict()
CYPV = dict()
ENVPV = dict()
for h in np.arange(0,49):
    pvloc[h] = np.array([])
    CYPV[h] = np.array([])
    ENVPV[h] = np.array([])

size = 1.5
#fig, axes = plt.subplots(1,3, subplot_kw=dict(projection=ccrs.PlateCarree()),sharex=True,sharey=True)
#plt.subplots_adjust(left=0.1,bottom=None,top=None,right=None,hspace=0,wspace=0)
#axes = axes.flatten()

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(nrows=3,ncols=1)
axes = []
for k in range(1):
    for l in range(3):
        axes.append(fig.add_subplot(gs[l,k],projection=ccrs.PlateCarree()))

hzetab = 6

CYM = np.array([])
for q,d in enumerate(dipv.keys()):
    ac[d] = dict()
    pressuredi[d] = dict()
    for h in np.arange(-48,49):
        ac[d][h] = np.array([])
        pressuredi[d][h] = np.array([])

    mon = MON[q][-9:-4:]
    ids = int(d[-3:])
    if (locdata[mon][ids]['hzeta'][0]>(-1* hzetab)):
        continue

    OL = PVdata['rawdata'][d]['OL']    
    PV = PVdata['rawdata'][d]['PV']
    pre = PVdata['rawdata'][d]['P']
    i = np.where((PV[:,0]>=0.75))[0]# & (pre[:,0]<=925))[0]
    if len(i)<200:
        continue
    xx = np.where(pre[:,0]<=925)[0]
    cycpvc = dipv[d][c]['deltaPV']
    envpvc = dipv[d][e]['deltaPV']

#    cycpvc[i,:-1] = dipv[d][c]['deltaPV'][i,:-1] - dipv[d][c]['deltaPV'][i,1:]
#    envpvc[i,:-1] = dipv[d][e]['deltaPV'][i,:-1] - dipv[d][e]['deltaPV'][i,1:]

    for h in np.arange(0,49):
        pvloc[h] = np.append(pvloc[h],PV[i,h])
        CYPV[h] = np.append(CYPV[h],cycpvc[i,h])
        ENVPV[h] = np.append(ENVPV[h],envpvc[i,h])

#    for we in i:
#        cyid = np.where(dit[d][c][we]==0)[0][0]
#        tn = np.flip(np.arange(-48,1))-np.flip(np.arange(-48,1))[cyid]
#        for a,b in enumerate(tn):
#            ac[d][b] = np.append(ac[d][b],PV[we,a])
#            pressuredi[d][b] = np.append(pressuredi[d][b],pre[we,a])

    pvend = PV[i,0]
    pvstart = PV[i,-1]
    ol = np.append(ol,OL[i,0])
    ol2 = np.append(ol2,np.mean(OL[i,0]))
    PVstart = np.append(PVstart,pvstart)
    PVend = np.append(PVend,pvend)

    cypv = dipv[d][c]['deltaPV'][i,0]
    enpv = dipv[d][e]['deltaPV'][i,0]
    CYM = np.append(CYM,np.mean(cypv))
#    print(d,len(i),len(pre[xx,0]),np.mean(pvstart/pvend),np.mean(cypv/pvend),np.mean(enpv/pvend),np.max(locdata[mon][ids]['zeta']),len(i)/len(pre[xx,0]))
    adv = np.append(adv,(pvstart)/pvend)
    cyc = np.append(cyc,cypv/pvend)
    env = np.append(env,enpv/pvend)
#    if np.mean(enpv/pvend)>0.3:
#        print(d,np.mean(pvstart/pvend),np.mean(cypv/pvend),np.mean(enpv/pvend))
#    if np.mean(enpv)/np.mean(pvend)>0.3:
#         print(d,np.mean(pvstart/pvend),np.mean(cypv/pvend),np.mean(enpv/pvend),np.mean(enpv)/np.mean(pvend))
    if np.mean(cypv/pvend)>0.6:
        axes[0].scatter(np.mean(PVdata['rawdata'][d]['lon'][:,0]),np.mean(PVdata['rawdata'][d]['lat'][:,0]),color='k',s=size)
##
    if np.mean(enpv/pvend)>0.6:
        axes[1].scatter(np.mean(PVdata['rawdata'][d]['lon'][:,0]),np.mean(PVdata['rawdata'][d]['lat'][:,0]),color='k',s=size)
##
    if np.mean(pvstart/pvend)>0.6:
        axes[2].scatter(np.mean(PVdata['rawdata'][d]['lon'][:,0]),np.mean(PVdata['rawdata'][d]['lat'][:,0]),color='k',s=size)
##
#    if np.mean(enpv/pvend)>0.5:
#        axes[3].scatter(np.mean(PVdata['rawdata'][d]['lon'][:,0]),np.mean(PVdata['rawdata'][d]['lat'][:,0]),color='k',s=size)
###
#    if np.mean(cypv/pvend)>0.6:
###        cycounter +=1
#        axes[4].scatter(np.mean(PVdata['rawdata'][d]['lon'][:,0]),np.mean(PVdata['rawdata'][d]['lat'][:,0]),color='k',s=size)
#    if np.mean(enpv/pvend)>0.6:
#        axes[5].scatter(np.mean(PVdata['rawdata'][d]['lon'][:,0]),np.mean(PVdata['rawdata'][d]['lat'][:,0]),color='k',s=size)
##        envcounter+=1
#    if np.mean(cypv/pvend)>0.75:
#        axes[6].scatter(np.mean(PVdata['rawdata'][d]['lon'][:,0]),np.mean(PVdata['rawdata'][d]['lat'][:,0]),color='k',s=size)
##        cycounter2+=1
#    if np.mean(enpv/pvend)>0.75:
#        axes[7].scatter(np.mean(PVdata['rawdata'][d]['lon'][:,0]),np.mean(PVdata['rawdata'][d]['lat'][:,0]),color='k',s=size)

    ct +=1
if CT=='MED':
    minpltlonc = -10
    maxpltlonc = 45
    minpltlatc = 25
    maxpltlatc = 50
    steps = 5

if CT=='ETA':
    minpltlonc = -90
    maxpltlonc = 50
    minpltlatc = 0
    maxpltlatc = 90
    steps = 10

lonticks=np.arange(minpltlonc, maxpltlonc+1,steps*2)
latticks=np.arange(minpltlatc, maxpltlatc+1,steps*2)    

#lab = ['40% cyc', '40% env','50% cyc', '50% env','60% cyc','60%env','75% cyc','75% env']#,'90% cyc','90% env']
lab = ['60% cyc','60% env', '60% adv']
labels = labels = ['a)','b)','c)','d)','e)','f)','g)','h)']
for q, ax in enumerate(axes):
#    ax.coastlines()
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)

    ax.set_aspect('auto')
    ax.set_xticks(lonticks, crs=ccrs.PlateCarree())
    ax.set_yticks(latticks, crs=ccrs.PlateCarree())
    if q%2==0:
        ax.set_yticklabels(labels=latticks,fontsize=10)
        ax.yaxis.set_major_formatter(LatitudeFormatter())
    if q==6 or q==7:
        ax.set_xticklabels(labels=lonticks,fontsize=10)
        ax.xaxis.set_major_formatter(LongitudeFormatter())

    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc])
    ax.text(0.45, 0.95, lab[q], transform=ax.transAxes,fontsize=8,va='top')
    ax.text(0.06, 0.85, labels[q], transform=ax.transAxes,fontsize=12, fontweight='bold',va='top')

plt.subplots_adjust(left=0.1,bottom=None,top=None,right=0.6,hspace=0,wspace=0)
fig.savefig('/home/ascherrmann/010-IFS/' + CT + '-cyclones-colored-contribution-t-%01d-IFS.png'%hzetab,dpi=300,bbox_inches="tight")
plt.close('all')

boxpv = []
tsc = []
cycbox = []
envbox = []
for h in np.flip(np.arange(0,49)):
    cycbox.append(np.sort(CYPV[h]))
    envbox.append(np.sort(ENVPV[h]))
    pvloc[h] = np.sort(pvloc[h])
    boxpv.append(pvloc[h])

tscx = np.array([])
#tmpd = dict()
#pred = dict()
#for h in np.arange(-48,49):
#    tmpd[h] = np.array([])
#    pred[h] = np.array([])
#for h in np.arange(-48,49):
#    for q,d in enumerate(dipv.keys()):
#        tmpd[h] = np.append(tmpd[h],ac[d][h])
#        pred[h] = np.append(pred[h],pressuredi[d][h])
#
#ntraj = np.array([])
#prec = np.array([])
#for h in tmpd.keys():
#    if len(tmpd[h]>10):
#        tscx = np.append(tscx,h)
#        tsc.append(tmpd[h])
#        prec = np.append(prec,np.mean(pred[h]))
#        ntraj = np.append(ntraj,len(tmpd[h]))

print(np.mean(adv),np.mean(cyc),np.mean(env),ct)
#print(np.mean(ol),ol2)

#pload = '/home/ascherrmann/TT/use/'
#
#traj2 = np.array([])
#MON2 = np.array([])
#for d in os.listdir(pload):
#    if d.startswith('trajectories-mature-'):
#        traj2 = np.append(traj2,d)
##
#for kk in MON:
#    for tt in traj2[:]:
#        if kk[-31:-10]==tt[-25:-4]:
#            MON2 = np.append(MON2,kk)
##            
#f = open(pload + 'PV-data-4days-MEDdPSP-100-ZB-800PVedge-0.3.txt','rb')
#PVdata = pickle.load(f)
#f.close()
##
#dipv = PVdata['dipv']
##adv = np.array([])
##cyc = np.array([])
##env = np.array([])
##
#
#PVstart2 = np.array([])
#PVend2 = np.array([])
#for q,d in enumerate(dipv.keys()):
#    mon = MON2[q][-9:-4:]
#    ids = int(d[-3:])
#    if (locdata[mon][ids]['hzeta'][0]>-6):
#        continue
#    PV = PVdata['rawdata'][d]['PV']
#    i = np.where(PV[:,0]>=0.75)[0]
#    pvend = PV[i,0]
#    pvstart = PV[i,-1]
#
#    PVstart2 = np.append(PVstart2,pvstart)
#    PVend2 = np.append(PVend2,pvend)
#
#    cypv = dipv[d][c]['deltaPV'][i,0]
#    enpv = dipv[d][e]['deltaPV'][i,0]
##    print(np.mean(pvstart/pvend),np.mean(cypv/pvend),np.mean(enpv/pvend),np.max(locdata[mon][ids]['zeta']))
#    adv = np.append(adv,(pvstart)/pvend)
#    cyc = np.append(cyc,cypv/pvend)
#    env = np.append(env,enpv/pvend)
#
#
#PVends = np.append(np.arange(0.75,2.01,0.125),1000)
#xx = np.array([])
#y = dict()
#y2 = dict()
#for q,k in enumerate(PVends[:-1]):
#    ids = helper.where_greater_smaller(PVend,k,PVends[q+1])
#    xx = np.append(xx,k)
#    y[k] = np.sort(PVstart[ids])
##
#    ids = helper.where_greater_smaller(PVend2,k,PVends[q+1])
#    y2[k] = np.sort(PVstart2[ids]) 
##
#fig,ax = plt.subplots()
##
#data = []
#data2 = []
##
#for k in xx[:]:
#    data.append(y[k])
#    data2.append(y2[k])
##
#ax.set_ylabel(r'PV$_{start}$ [PVU]')
#ax.set_xlabel(r'PV$_{end}$ [PVU]')
#ax.set_ylim(-0.25,1.25)
#
flier = dict(marker='+',markerfacecolor='grey',markersize=1,linestyle=' ',markeredgecolor='grey')
meanline = dict(linestyle='-',linewidth=1,color='red')
#
meanline2 = dict(linestyle='-',linewidth=1,color='dodgerblue')
capprops = dict(linestyle='-',linewidth=1,color='grey')
medianprops = dict(linestyle='-',linewidth=1,color='grey')
medianprops1= dict(linestyle='-',linewidth=1,color='black')
boxprops = dict(linestyle='-',linewidth=1.,color='slategrey')
whiskerprops= dict(linestyle='-',linewidth=1,color='grey')
#
#
#bp = ax.boxplot(data,whis=(10,90),labels=xx,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False)#positions=range(len(xx)),flierprops=flier)
#bp2 = ax.boxplot(data2,whis=(10,90),labels=xx,flierprops=flier,meanprops=meanline2,meanline=True,showmeans=True,showbox=True,showcaps=True,showfliers=False,medianprops=medianprops,capprops=capprops,whiskerprops=whiskerprops,boxprops=boxprops)
#ax.set_xticks(ticks=range(1,len(xx)+1))
#ax.set_xticklabels(labels=xx)#['','0.75','0.875','1.0','1.125','1.25','1.375','1.5','1.625','1.75','1.875','2.0',''])
##
#fig.savefig('/home/ascherrmann/010-IFS/boxwis-2-4days-6hprio-PV-start-end.png',dpi=300,bbox_inches="tight")
#plt.close('all')
#
#
print(np.mean(boxpv[-1]),np.mean(boxpv[0]))
fig,ax = plt.subplots()
ax.set_ylabel(r'PV [PVU]')
ax.set_xlabel(r'time to mature stage [h]')
ax.set_ylim(-.25,2.0)
ax.set_xlim(0,50)
t = np.arange(-48,1)
bp = ax.boxplot(boxpv,whis=(10,90),labels=t,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops1)
ax.set_xticks(ticks=range(1,len(t)+1,6))
ax.tick_params(labelright=False,right=True)
ax.set_xticklabels(labels=t[0::6])
#fig.savefig('/home/ascherrmann/010-IFS/boxwis-high-PV-t-%01dh-IFS.png'%hzetab,dpi=300,bbox_inches="tight")
plt.close('all')

fig,ax = plt.subplots()
ax.set_ylabel(r'PV [PVU]')
ax.set_xlabel(r'time to mature stage [h]')
ax.set_ylim(-.750,1.5)
ax.set_xlim(0,50)
t = np.arange(-48,1)
bp = ax.boxplot(cycbox,whis=(10,90),labels=t,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops1)
bp2 = ax.boxplot(envbox,whis=(10,90),labels=t,flierprops=flier,meanprops=meanline2,meanline=True,showmeans=True,showbox=True,showcaps=True,showfliers=False,medianprops=medianprops,capprops=capprops,whiskerprops=whiskerprops,boxprops=boxprops)
#
ax.set_xticks(ticks=range(1,len(t)+1,6))
ax.tick_params(labelright=False,right=True)
ax.set_xticklabels(labels=t[0::6])
print(np.mean(cycbox[-1]),np.mean(envbox[-1]))
#fig.savefig('/home/ascherrmann/010-IFS/boxwis-cyc-env-high-PV-t-%01dh-IFS.png'%hzetab,dpi=300,bbox_inches="tight")
plt.close('all')
#
#fig,ax = plt.subplots()
#ax2 = ax.twinx()
#ax.set_ylabel(r'PV [PVU]')
#ax.set_xlabel(r'time entering the cyclone [h]')
#ax.set_ylim(-.25,2.0)
#ax.set_xlim(0,len(tscx)+1)
#t = tscx
#zid = np.where(t==0)[0]
#ax.axvline(zid+1,color='k',alpha=0.4)
#ax.axvline(zid+1-6,color='k',alpha=0.4)
#ax.axvline(zid+1+6,color='k',alpha=0.4)
#ax2.plot(np.arange(1,len(t)+1),ntraj,color='blue',alpha=0.6)
#ax2.set_ylabel('number of trajectories')
#ax2.set_yscale('log')
#firsttick = np.where(t%6==0)[0][0]
#atick = np.arange(1,len(t))
##
##tick = np.append(np.flip(np.flip(np.arange(
#bp = ax.boxplot(tsc,whis=(10,90),labels=t,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False)
#ax.set_xticks(ticks=np.arange(atick[firsttick],len(tscx),6))
##ax.tick_params(labelright=False,right=True)
#ax.set_xticklabels(labels=np.arange(t[firsttick],t[-1]+0.000001,6).astype(int))
#fig.savefig('/home/ascherrmann/010-IFS/boxwis-PV-enter-cyclone.png',dpi=300,bbox_inches="tight")
#plt.close('all')
