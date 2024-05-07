import numpy as np
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
sys.path.append('/home/raphaelp/phd/scripts/basics/')
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
import helper
import matplotlib
import matplotlib.pyplot as plt
import pickle

pload = '/home/ascherrmann/010-IFS/ctraj/MED/use/'

CT = 'MED'

f = open(pload + 'PV-data-' + CT + 'dPSP-100-ZB-800PVedge-0.3-400-correct-distance.txt','rb')
data = pickle.load(f)
f.close()
labs = helper.traced_vars_IFS()
dipv = data['dipv']
dit = data['dit']


f = open('/home/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','rb')
locdata =  pickle.load(f)
f.close()

MON = np.array([])
for d in os.listdir(pload[:-4]):
    if(d.startswith('traend-')):
        MON = np.append(MON,d)
MON = np.sort(MON)

resev = dict()
cycres = dict()
envres =dict()

for h in range(0,49):
    resev[h] = np.array([])
    cycres[h] = np.array([])
    envres[h]= np.array([])

LON = np.arange(-180,180.1,0.4)
LAT = np.arange(0,90.1,0.4)

boxres = []
cycboxres = []
envboxres = []
avcyclone = []

summedPVs = dict()
fig, axes = plt.subplots()
axes.plot([],[],ls='-',color='grey')
axes.plot([],[],ls='-',color='k')
axes.plot([],[],ls='-',color='k')
axes.plot([],[],ls=':',color='k')

dlon = np.arange(-3,3.1,0.4)
dlat = dlon


fig2,ax = plt.subplots()
colors=['k','grey']
for q,date in enumerate(data['rawdata'].keys()):
    if q==0:
        for pvr in helper.traced_vars_IFS()[8:]:
            summedPVs[pvr] = np.zeros(2)
#   if date=='20171214_02-073' or date=='20180417_02-076':
#    if date=='20180303_09-017':
#        figs,axes = plt.subplots(1,1,subplot_kw=dict(projection=ccrs.PlateCarree()))
#        axes.coastlines()
#        axes.scatter(np.mean(datadi[date]['lon'][:,0]),np.mean(datadi[date]['lat'][:,0]),color='k')
#        minpltlonc = -10
#        maxpltlonc = 45
#        minpltlatc = 25
#        maxpltlatc = 50
#        steps = 5
#    
#        lonticks=np.arange(minpltlonc, maxpltlonc,steps)
#        latticks=np.arange(minpltlatc, maxpltlatc,steps)
#        
#        axes.set_xticks(lonticks, crs=ccrs.PlateCarree());
#        axes.set_yticks(latticks, crs=ccrs.PlateCarree());
#        axes.set_xticklabels(labels=lonticks,fontsize=10)
#        axes.set_yticklabels(labels=latticks,fontsize=10)
#        
#        axes.xaxis.set_major_formatter(LongitudeFormatter())
#        axes.yaxis.set_major_formatter(LatitudeFormatter())
#
#        axes.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
#    if date=='20180417_02-076':
#        axes.scatter(np.mean(datadi[date]['lon'][:,0]),np.mean(datadi[date]['lat'][:,0]),color='r')
#
#        figs.savefig('/home/ascherrmann/010-IFS/test.png',dpi=300,bbox_inches="tight")
#        plt.close('all')
    counter = np.zeros((len(dlon),len(dlat)))
    
    ids = int(date[-3:])
    mon = MON[q][-9:-4:]
    if date!='20180619_03-111' and date!='20171214_02-073':
        continue
    if date=='20180619_03-111':
        ls = ':'
        color='blue'
    else:
        ls = '-'
        color='k'
    #if (locdata[mon][ids]['hzeta'][0]>-24):
    #    continue
    PVf = data['rawdata'][date]['PV']
    datadi = data['rawdata']
    idp = np.where(PVf[:,0]>=0.75)[0]
    CYLON = np.mean(LON[locdata[mon][ids]['clon'][abs(locdata[mon][ids]['hzeta'][0]).astype(int)]])
    CYLAT = np.mean(LAT[locdata[mon][ids]['clat'][abs(locdata[mon][ids]['hzeta'][0]).astype(int)].astype(int)])
    
    #ax.scatter(datadi[date]['lon'][idp,0]-CYLON,datadi[date]['lat'][idp,0]-CYLAT,marker='.',color=color)
    for la,lo in zip(datadi[date]['lat'][idp,0]-CYLAT,datadi[date]['lon'][idp,0]-CYLON):
        ll = np.where(abs(dlat-la)==np.min(abs(dlat-la)))[0][0]
        lq = np.where(abs(dlon-lo)==np.min(abs(dlon-lo)))[0][0]
        counter[ll,lq]+=1
    cf = ax.contour(dlon,dlat,counter,colors=color,linewidths=1.5,levels=[5,10,15,20,25,30])
    ax.clabel(cf,inline=True,manual=True,fmt='%d',fontsize=10)
#    APVTOT = np.zeros(data['rawdata'][date]['PV'].shape)
#    for pv in labs[8:]:
#        APVTOT[:,1:] +=np.cumsum(data['rawdata'][date][pv][:,1:],axis=1)

#    res = (APVTOT[idp,:]) - (PVf[idp,:]*(-1.) + PVf[idp,0][:,None])
    res = (dipv[date]['env']['APVTOT'][idp,:] + dipv[date]['cyc']['APVTOT'][idp,:])-(PVf[idp,:]-PVf[idp,-1][:,None])
    PV = np.mean(PVf[idp,:],axis=0)

    datadi[date]['DELTAPV'] = np.zeros(datadi[date]['PV'].shape)
    datadi[date]['DELTAPV'][:,1:] = datadi[date]['PV'][:,:-1]-datadi[date]['PV'][:,1:]
    datadi[date]['RES'] = np.zeros(datadi[date]['PV'].shape)

    datadi[date]['PVRTOT'] = np.zeros(datadi[date]['PV'].shape)
    for pv in labs[8:]:
        datadi[date]['PVRTOT']+=datadi[date][pv]

    datadi[date]['RES'][:,:-1] = np.flip(np.cumsum(np.flip(datadi[date]['PVRTOT'][:,1:],axis=1),axis=1),axis=1)-np.flip(np.cumsum(np.flip(datadi[date]['DELTAPV'][:,1:],axis=1),axis=1),axis=1)
    pvpredict = np.mean(dipv[date]['cyc']['APVTOT'][idp,:] + dipv[date]['env']['APVTOT'][idp,:],axis=0)

    for h in range(0,49):
        resev[h] = np.append(resev[h],datadi[date]['RES'][idp,h])
        cycres[h] = np.append(cycres[h],datadi[date]['RES'][idp,h][dit[date]['cyc'][idp,h]!=0])
        envres[h] = np.append(envres[h],datadi[date]['RES'][idp,h][dit[date]['env'][idp,h]!=0])

    for pvr in helper.traced_vars_IFS()[8:]:
        summedPVs[pvr][0] += np.sum(abs(datadi[date][pvr][idp,:][dit[date]['cyc'][idp,:]!=0]))
        summedPVs[pvr][1] += np.sum(abs(datadi[date][pvr][idp,:][dit[date]['env'][idp,:]!=0]))

#    datadi[date]['RES2'] = np.zeros(datadi[date]['PV'].shape)
#    datadi[date]['RES2'][:,1:] = np.cumsum(datadi[date]['PVRTOT'][:,:-1],axis=1)-np.cumsum(datadi[date]['DELTAPV'][:,1:],axis=1)
#    datadi[date]['RES3'] = np.zeros(datadi[date]['PV'].shape)
#    datadi[date]['newPVR'] = np.zeros(datadi[date]['PV'].shape)
#    datadi[date]['newPVR'][:,1:] = (datadi[date]['PVRTOT'][:,1:] + datadi[date]['PVRTOT'][:,:-1])/2

#    datadi[date]['RES3'][:,1:] = np.cumsum(datadi[date]['newPVR'][:,1:],axis=1)-np.cumsum(datadi[date]['DELTAPV'][:,1:],axis=1)
#    datadi[date]['locres'] = np.zeros(datadi[date]['PV'].shape)
#    datadi[date]['locres'][:,1:] = datadi[date]['PVRTOT'][:,1:] - datadi[date]['DELTAPV'][:,1:]

    avcyclone.append(np.sort(datadi[date]['RES'][idp,0]))

    res = datadi[date]['RES']

    H = 48
    t = np.flip(np.arange(-48,1,1))
    xlab='time until mature stage [h]'
#    fig, axes = plt.subplots()
    
    axes.set_xlabel(xlab,fontsize=8)
    axes.set_ylabel('PV [PVU]')
    
    #for k in range(len(res)):
    #    axes.plot(t,res[k],color='black')
#    axes.plot(t,np.mean(res[idp],axis=0),color='black')
    axes.plot(t,PV,color='grey',linestyle=ls)
    axes.plot(t,pvpredict + PV[-1],color='k',linestyle=ls) 

#    axes.plot(t,np.mean(datadi[date]['RES2'][idp],axis=0),color='grey')
#    axes.plot(t,np.mean(datadi[date]['RES3'][idp],axis=0),color='blue')
axes.set_xticks(ticks=np.arange(-48,1,6))
#axes.set_yticks(ticks=np.arange(-0.5,1.8,0.1))
#axes.set_yticklabels(labels=np.array([-0.5,'','-0.3','',-0.1,'',0.1,'',0.3,'',0.5,'',0.7,'',0.9,'',1.1,'',1.3,'',1.5,'',1.7]))
axes.set_xlim(-48,0)
axes.set_ylim(0,1.8)
#axes.set_ylim(-0.2,1.3)
axes.tick_params(labelright=False,right=True)
name='PV-evol' + date + '.png'
name = 'PV-evol-combined.png'
axes.legend(['PV','estimated PV','20171214_02','20180619_03'])
fig.savefig('/home/ascherrmann/010-IFS/' + name,dpi=300,bbox_inches="tight")
print('saved')
plt.close(fig) 
ax.scatter(0,0,marker='o',color='red',s=20)
ax.set_xticks(ticks=np.arange(-3,3.1,0.5))
ax.set_yticks(ticks=np.arange(-3,3.1,0.5))
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.grid(True,zorder=0,linewidth=0.5)
fig2.savefig('/home/ascherrmann/010-IFS/' + 'traj-start-scat.png',dpi=300,bbox_inches="tight")
plt.close('all')



nc = summedPVs['PVRLS'][0].astype(int)
print(nc)
#nc = 14058
for pvr in helper.traced_vars_IFS()[8:]:
    print(pvr, summedPVs[pvr].astype(int)/nc)

for h in np.flip(np.arange(0,49)):
    boxres.append(np.sort(resev[h]))
    cycboxres.append(np.sort(cycres[h]))
    envboxres.append(np.sort(envres[h]))

flier = dict(marker='+',markerfacecolor='grey',markersize=1,linestyle=' ',markeredgecolor='grey')
meanline = dict(linestyle='-',linewidth=1,color='red')
meanline2 = dict(linestyle='-',linewidth=1,color='dodgerblue')
capprops = dict(linestyle='-',linewidth=1,color='grey')
medianprops = dict(linestyle='-',linewidth=1,color='magenta')
boxprops = dict(linestyle='-',linewidth=1.,color='slategrey')
whiskerprops= dict(linestyle='-',linewidth=1,color='grey')


fig,ax = plt.subplots()
ax.set_ylabel(r'residual [PVU]')
ax.set_xlabel(r'time to mature stage [h]')
ax.set_ylim(-2.5,2)
ax.set_xlim(0,50)
t = np.arange(-48,1)
bp = ax.boxplot(boxres,whis=(10,90),labels=t,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False)

ax.set_xticks(ticks=range(1,len(t)+1,6))
ax.tick_params(labelright=False,right=True)
ax.set_xticklabels(labels=t[0::6])
#fig.savefig('/home/ascherrmann/010-IFS/residual-boxwis-time-6h.png',dpi=300,bbox_inches="tight")
plt.close('all')

labs = np.array([])
for k in range(1,len(avcyclone)+1):
    labs = np.append(labs,' ')

fig,ax = plt.subplots()
ax.set_ylabel(r'residual [PVU]')
ax.set_xlabel(r'time to mature stage [h]')
ax.set_ylim(-2.5,2)
ax.set_xlim(0,50)
t = np.arange(-48,1)
bp = ax.boxplot(cycboxres,whis=(10,90),labels=t,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False)
bp2 = ax.boxplot(envboxres,whis=(10,90),labels=t,flierprops=flier,meanprops=meanline2,meanline=True,showmeans=True,showbox=True,showcaps=True,showfliers=False,medianprops=medianprops,capprops=capprops,whiskerprops=whiskerprops,boxprops=boxprops)

ax.set_xticks(ticks=range(1,len(t)+1,6))
ax.tick_params(labelright=False,right=True)
ax.set_xticklabels(labels=t[0::6])
#fig.savefig('/home/ascherrmann/010-IFS/residual-boxwis-cyc-env-time-6h.png',dpi=300,bbox_inches="tight")
plt.close('all')


fig,ax = plt.subplots()
ax.set_ylabel(r'residual [PVU]')
ax.set_ylim(-2.5,2)
ax.set_xlim(0,len(avcyclone)+1)
bp = ax.boxplot(avcyclone,whis=(10,90),labels=labs,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False)
ax.tick_params(labelright=False,right=True)

#fig.savefig('/home/ascherrmann/010-IFS/residual-cyclones-6h.png',dpi=300,bbox_inches="tight")
plt.close('all')
