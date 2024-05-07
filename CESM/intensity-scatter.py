import sys
sys.path.append('/home/ascherrmann/scripts/')
import wrfsims
import numpy as np
from netCDF4 import Dataset as ds
import matplotlib.pyplot as plt
import wrf
import pickle
import os

sim,at,med=wrfsims.cesm_ids()

tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'
ics='/atmosdyn2/ascherrmann/013-WRF-sim/ics/'
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
sea='DJF'

colors=['k','blue','yellow','orange','red']
x0,y0,x1,y1=70,30,181,101

ofsetfac=[0,0.5,1,2]
xof=[0,-8,8,0,0]
yof=[0,0,0,-8,8]
names = ['-0-km','west','east','south','north']
km=['-0-km','200','400','800']
period=['ERA5','2010','2040','2070','2100']
refx=dict()
refy=dict()

for perio in period:
    d = ds(ics + 'CESM-%s-%s-clim/met_em.d01.2000-12-01_00:00:00.nc'%(perio,sea),'r')
    u=np.squeeze(d.variables['UU'][:])
    p=np.squeeze(d.variables['PRES'][:])
    pl=np.where(p[:,0,0]==300*100)[0][0]
    u300roi=u[pl,y0:y1,x0:x1]
    ymax,xmax=np.where(u300roi==np.max(u300roi))[0][0],np.where(u300roi==np.max(u300roi))[1][0]

    xmax+=x0
    ymax+=y0
    refx[perio],refy[perio]=xmax,ymax
    

plt.close('all')
fig,ax = plt.subplots(figsize=(8,6))
ax.plot([],[],marker='o',ls='',color='grey')
ax.plot([],[],marker='s',ls='',color='grey')
for co,perio in zip(colors,period):
    ax.plot([],[],marker='o',ls='',color=co)

pvdi=dict()
atslpdi=dict()
medslpdi=dict()

for perio in period:
    pvdi[perio]=[]
    atslpdi[perio]=[]
    medslpdi[perio]=[]


figg,axx = plt.subplots(figsize=(8,6))

if os.path.isfile('/atmosdyn2/ascherrmann/015-CESM-WRF/CESM-slp-PV-data.txt'):
    f = open('/atmosdyn2/ascherrmann/015-CESM-WRF/CESM-slp-PV-data.txt','rb')
    save = pickle.load(f)    
    f.close()

    pvdi=save['pv']
    atslpdi=save['at']
    medslpdi=save['med']

if not os.path.isfile('/atmosdyn2/ascherrmann/015-CESM-WRF/CESM-slp-PV-data.txt'):
    for si,a,m in zip(sim,at,med):
    #    if not '2070' in si and not '2100' in si:
    #        continue
        if si[-4:]=='clim':
            continue
        a=np.array(a)
        m=np.array(m)
        if np.any(m==None):
            continue
            
        # position ofset
        for xo,yo,pos in zip(xof,yof,names):
            if pos in si:
                print(si,'in')
                break
        # distance factor
        for offac,k in zip(ofsetfac,km):
            if k in si:
                break    
        # get correct position
        for co,perio in zip(colors,period):
            if perio in si:    
                xxmax,xymax=int(refx[perio]+xo*offac),int(refy[perio]+yo*offac)
                break
        # load
        ic = ds(dwrf + si + '/wrfout_d01_2000-12-01_00:00:00')
        tra = np.loadtxt(tracks + si + '-new-tracks.txt')
    
        # store
        t = tra[:,0]
        tlon,tlat = tra[:,1],tra[:,2]
        slp = tra[:,3]
        IDs = tra[:,-1]
    
        PV = wrf.getvar(ic,'pvo')
        p = wrf.getvar(ic,'pressure')
    
        pv = wrf.interplevel(PV,p,300,meta=False)
        
    #    print(perio,refx[perio],refy[perio],xxmax,xymax)
        maxpv=pv[xymax,xxmax]
        
        loc = np.where(IDs==2)[0]
        slpmin = np.min(slp[loc])
        loc = np.where(IDs==1)[0]
        aminslp = np.min(slp[loc])
        if aminslp>1000:
            print(aminslp,si)
    
        ax.scatter(maxpv,aminslp,color=co,marker='s')
        ax.scatter(maxpv,slpmin,color=co,marker='o')
        axx.scatter(aminslp,slpmin,color=co)
    
        pvdi[perio].append(maxpv)
        atslpdi[perio].append(aminslp)
        medslpdi[perio].append(slpmin)


if os.path.isfile('/atmosdyn2/ascherrmann/015-CESM-WRF/CESM-slp-PV-data.txt'):
    for co,perio in zip(colors,period):
        ax.scatter(pvdi[perio],atslpdi[perio],color=co,marker='s')
        ax.scatter(pvdi[perio],medslpdi[perio],color=co,marker='o')
        axx.scatter(atslpdi[perio],medslpdi[perio],color=co)

save=dict()
save['pv']=pvdi
save['at']=atslpdi
save['med']=medslpdi

f=open('/atmosdyn2/ascherrmann/015-CESM-WRF/CESM-slp-PV-data.txt','wb')
pickle.dump(save,f)
f.close()


sim,at,med=wrfsims.upper_ano_only()
if not os.path.isfile('/atmosdyn2/ascherrmann/015-CESM-WRF/paper-ERA5-slp-pv-data.txt'):
    e5pv=[]
    e5atslp=[]
    e5medslp=[]
    for si,a,m in zip(sim,at,med):
        if si[-4:]=='clim' or 'nested' in si or '0.5' in si or '0.3' in si or 'not' in si or '0.9' in si or '1.1' in si or '1.7' in si or '2.8' in si or 'DJF' not in si or 'check' in si:
            continue

        a=np.array(a)
        m=np.array(m)

        # position ofset
        inn = 0
        for xo,yo,pos in zip(xof,yof,names):
            if pos in si:
                print(si,'in')
                inn=1
                break
        if inn==0:
            xo,yo=0,0
        # distance factor
        for offac,k in zip(ofsetfac,km):
            if k in si:
                break
        print('new',si)

        # get correct position
        xxmax,xymax=int(93+xo*offac),int(55+yo*offac)
        # load
        ic = ds(dwrf + si + '/wrfout_d01_2000-12-01_00:00:00')
        tra = np.loadtxt(tracks + si + '-new-tracks.txt')

        # store
        t = tra[:,0]
        tlon,tlat = tra[:,1],tra[:,2]
        slp = tra[:,3]
        IDs = tra[:,-1]

        PV = wrf.getvar(ic,'pvo')
        p = wrf.getvar(ic,'pressure')

        pv = wrf.interplevel(PV,p,300,meta=False)

        maxpv=pv[xymax,xxmax]

        loc = np.where(IDs==2)[0]
        slpmin = np.min(slp[loc])
        loc = np.where(IDs==1)[0]
        aminslp = np.min(slp[loc])
        e5pv.append(maxpv)
        e5atslp.append(aminslp)
        e5medslp.append(slpmin)

    era5di=dict()
    era5di['pv']=e5pv
    era5di['at']=e5atslp
    era5di['med']=e5medslp

    f=open('/atmosdyn2/ascherrmann/015-CESM-WRF/paper-ERA5-slp-pv-data.txt','wb')
    pickle.dump(era5di,f)
    f.close()


f=open('/atmosdyn2/ascherrmann/015-CESM-WRF/paper-ERA5-slp-pv-data.txt','rb')
e5dat = pickle.load(f)
f.close()


flier = dict(marker='+',markerfacecolor='grey',markersize=1,linestyle=' ',markeredgecolor='grey')
meanline = dict(linestyle='-',linewidth=1,color='red')
meanline2 = dict(linestyle='-',linewidth=1,color='dodgerblue')
capprops = dict(linestyle='-',linewidth=1,color='grey')
medianprops = dict(linestyle='-',linewidth=1,color='black')
boxprops = dict(linestyle='-',linewidth=1.,color='slategrey')
whiskerprops= dict(linestyle='-',linewidth=1,color='grey')


leg=np.append(np.array(['Mediterranean','North Atlantic']),np.array(period))
ax.legend(leg,loc='upper right')
fig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/images/intensity-scatter-PV-CESM.png',dpi=300,bbox_inches='tight')
figg.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/images/intensity-scatter-SLP-CESM.png',dpi=300,bbox_inches='tight')



sim,at,med=wrfsims.cesm_jet_ids()
jetmedslp = []
jetatslp = []
for si,a,m in zip(sim,at,med):

        a=np.array(a)
        m=np.array(m)
        if np.any(m==None):
            continue
        tra = np.loadtxt(tracks + si + '-new-tracks.txt')

        # store
        t = tra[:,0]
        slp = tra[:,3]
        IDs = tra[:,-1]

        loc = np.where(IDs==2)[0]
        slpmin = np.min(slp[loc])
        jetmedslp.append(slpmin)

        loc = np.where(IDs==1)[0]
        slpmin = np.min(slp[loc])
        jetatslp.append(slpmin)

meanbox=[e5dat['med']]
for perio in period:
    meanbox.append(medslpdi[perio])

#meanbox.append(jetmedslp)


labels=np.array(['ERA5 42y','ERA5 PD','CESM PD','CESM SC','CESM MC','CESM EC'])#,'CESM EC'])
figgg,axxx=plt.subplots(figsize=(9,6))
bp = axxx.boxplot(meanbox,whis=(10,90),labels=labels,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops)
for kkk in range(len(meanbox)):
    axxx.text(kkk+1,1017,'%d'%len(meanbox[kkk]),horizontalalignment='center')

axxx.set_ylabel(r'minimum SLP [hPa]')
#axxx.set_xlim(0,8)
axxx.set_xlim(0,7)
axxx.set_ylim(990,1020)
axxx.set_xticklabels(labels=labels)

#plt.xticks(rotation=90)
figgg.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/images/SLP-box-whiskers.png',dpi=300,bbox_inches='tight')

plt.close('all')

meanbox=[e5dat['at']]
for perio in period:
    meanbox.append(atslpdi[perio])

#meanbox.append(jetatslp)

fig,axxx=plt.subplots()
bp = axxx.boxplot(meanbox,whis=(10,90),labels=labels,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops)
axxx.set_ylabel(r'minimum SLP [hPa]')
axxx.set_xlim(0,8)
axxx.set_ylim(960,1010)
axxx.set_xticklabels(labels=labels)

#plt.xticks(rotation=90)
fig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/images/SLP-Atlantic-box-whiskers.png',dpi=300,bbox_inches='tight')




f = open('/atmosdyn2/ascherrmann/015-CESM-WRF/CESM-delta-SLP-track.txt','rb')
save = pickle.load(f)
f.close()

fig,ax=plt.subplots()
meanbox=[]
for perio in list(save.keys())[:-1]:
    meanbox.append(save[perio])

print(len(meanbox))
bp = ax.boxplot(meanbox,whis=(10,90),labels=labels[1:],flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops)
ax.set_ylabel(r'min SLP - clim. SLP [hPa]')
ax.set_xlim(0,6)
ax.set_ylim(-25,-5)
ax.set_xticklabels(labels=labels[1:])

#plt.xticks(rotation=90)
fig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/images/delta-SLP-box-whiskers.png',dpi=300,bbox_inches='tight')


