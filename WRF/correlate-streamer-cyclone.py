import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")
import pickle
pcd = '/atmosdyn2/ascherrmann/011-all-ERA5/data/pandas-basic-data-all-deep-over-sea-12h.csv'

cd = pd.read_csv(pcd)
cd = cd.loc[cd['reg']=='MED']
SLP = cd['minSLP'].values
IDs = cd['ID'].values
lon = cd['lon'].values
lat = cd['lat'].values

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
ps2 = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
dfD = pd.read_csv(ps2 + 'DJF-intense-cyclones.csv')
dfS = pd.read_csv(ps2 + 'SON-intense-cyclones.csv')

ncD = np.array([])
ncS = np.array([])

#### get the 6 most abundant clusters in terms of number of cyclones
for cl in np.unique(dfD['region'].values):
    ncD = np.append(ncD,len(np.where(dfD['region'].values==cl)[0]))

for cl in np.unique(dfS['region'].values):
    ncS = np.append(ncS,len(np.where(dfS['region'].values==cl)[0]))

DJFcluster = np.unique(dfD['region'].values)[np.argsort(ncD)[-6:]]
SONcluster = np.unique(dfS['region'].values)[np.argsort(ncS)[-6:]]
ncD = ncD[np.argsort(ncD)[-6:]]
ncS = ncS[np.argsort(ncS)[-6:]]


#var=['time','lon','lat','area','avPV','integratedPV','maxPV','max10PV','max20PV','maxPVlon','maxPVlat']
#ylab=['time','lon','lat','area [npoints]','av. PV anomaly [PVU]','integrated PV [PVU]','maxPV [PVU]','max10PV [PVU]','max20PV [PVU]']
#for plv in ['area','avPV','integratedPV','maxPV','max10PV','max20PV']:
pre = ['300','350','400','450']
col = ['navy','cyan','orange','red']

var = ['time','streamerID','overlappingarea','steamerarea','avPV','maxPV','overlapavPV','overlapmaxPV']
ylab=['time','streamerID','overlapping area [points]', 'steamerarea [points]', 'av. PV anomaly [PVU]', 'max. PV anomaly [PVU]','av. overlap PV [PVU]', 'max.overlap PV [PVU]']


#for plv in ['overlappingarea','steamerarea','avPV','maxPV','overlapavPV','overlapmaxPV']:
#    fig,ax = plt.subplots()
#    i = np.where(np.array(var)==plv)[0][0]
#    yl = ylab[i]
#    n=0
#    slps = np.array([])
#    va = np.array([])
#    vadi =dict()
#    Slp = dict()
#    lens = np.array([])
#    for pr in pre:
#        vadi[pr]=np.array([])
#        Slp[pr] = np.array([])
#    for dirs in os.listdir(ps):
#        if dirs[-1]!='c' and dirs[-1]!='t':
#            di = dirs +'/'
#            ID = int(dirs)
#
#            for pr,cl in zip(pre,col):
#                strack = np.loadtxt(ps + di + 'overlapping-streamer-tracks-%s.txt'%pr,skiprows=1)
#                if strack.size==0:
#                    continue
#                strack = strack.reshape(-1,8)
#                if not np.any(strack[:,0]==0):
#                    continue
#                t0 = np.where(strack[:,0]==0)[0][0]
#                plotvar=strack[t0,i]
#                slp=SLP[np.where(IDs==ID)[0][0]]
#                vadi[pr] = np.append(vadi[pr],plotvar)
#                ax.scatter(slp,plotvar,color=cl)
#                Slp[pr] = np.append(Slp[pr],slp)
#    
#    for pr in pre:
#        print(pr,plv,pearsonr(Slp[pr],vadi[pr]))
#        lens=np.append(lens,len(Slp[pr]))
#
#    ax.set_xlabel('SLP [hPa]')
#    ax.set_ylabel(yl)
#    fig.savefig(pi + 'streamer-slp-' + plv + '.png',dpi=300,bbox_inches='tight')
#    plt.close('all')
    

#fig,axs = plt.subplots(2,2,sharex=True,sharey=True)
#plt.subplots_adjust(wspace=0,hspace=0)
#axs = axs.flatten()

f = open(ps + 'overlapp-mature-individual-counts.txt','rb')
data = pickle.load(f)
f.close()

trange = np.arange(-168,1,3)
IDS = np.array(list(data.keys()))
tranges = []
for ID in IDS:
    tranges.append(np.array(list(data[ID]['300'].keys())))

counter = dict()
levels = [np.arange(0.2,1.1,0.1),np.arange(0.2,1.1,0.1),np.arange(0.1,1.1,0.05),np.arange(0.05,1.1,0.05)]
overlap = dict()
for t in trange:
    overlap[t] = np.zeros((4,41,41))
    counter[t] = np.zeros(4)
    for ID in IDS:
        for w,pr in enumerate(pre):
            try:
                overlap[t][w]+= data[ID][pr][t]
                counter[t][w]+=1
            except:
                continue

if not os.path.isdir(pi + 'overlap-streamer/'):
        os.mkdir(pi + 'overlap-streamer/')

for t in counter.keys():
  if not np.all(counter[t]==0):
    fig,axs = plt.subplots(2,2,sharex=True,sharey=True)
    plt.subplots_adjust(wspace=0,hspace=0)
    axs = axs.flatten()
    for pr,q,ax,lvl in zip(pre,range(len(axs)),axs,levels):
        if counter[t][q]==0:
            continue
        h = ax.contour(np.arange(-10,10.5,0.5),np.arange(-10,10.5,0.5),overlap[t][q]/counter[t][q],colors='k',linewidths=1,levels=lvl)
        plt.clabel(h, inline=1, fontsize=8, fmt='%.2f')
        ax.scatter(0,0,marker='o',color='r')
        ax.text(0.05,0.95,'%s hPa'%pr,fontsize=6,transform=ax.transAxes)
        ax.text(0.45,0.95,'%d h'%t,fontsize=6,transform=ax.transAxes)
    fig.savefig(pi + 'overlap-streamer/' + 'streamer-overlap-%d.png'%t,dpi=300,bbox_inches='tight')
    plt.close('all')

trange = np.arange(-168,1,3)
for clus in DJFcluster:
    if not os.path.isdir(pi + 'overlap-streamer/' + clus):
        os.mkdir(pi + 'overlap-streamer/' + clus)

    overlap = dict()
    counter = dict()
    tmp = dfD.iloc[dfD['region'].values==clus]
    ids = tmp['ID'].values
    for t in trange:
        overlap[t] = np.zeros((4,41,41))
        counter[t] = np.zeros(4)
        for ID in ids:
            for w,pr in enumerate(pre):
                try:
                    overlap[t][w]+= data[ID][pr][t]
                    counter[t][w]+=1
                except:
                    continue


    for t in counter.keys():
      if not np.all(counter[t]==0):
        fig,axs = plt.subplots(2,2,sharex=True,sharey=True)
        plt.subplots_adjust(wspace=0,hspace=0)
        axs = axs.flatten()
        for pr,q,ax,lvl in zip(pre,range(len(axs)),axs,levels):
            if counter[t][q]==0:
                continue
            h = ax.contour(np.arange(-10,10.5,0.5),np.arange(-10,10.5,0.5),overlap[t][q]/counter[t][q],colors='k',linewidths=1,levels=lvl)
            plt.clabel(h, inline=1, fontsize=8, fmt='%.2f')
            ax.scatter(0,0,marker='o',color='r')
            ax.text(0.05,0.95,'%s hPa'%pr,fontsize=6,transform=ax.transAxes)
            ax.text(0.45,0.95,'%d h'%t,fontsize=6,transform=ax.transAxes)
        fig.savefig(pi + 'overlap-streamer/' + clus + '/' + clus + '-streamer-overlap-%d.png'%t,dpi=300,bbox_inches='tight')
        plt.close('all')




#for dirs in os.listdir(ps):
#    if dirs[-1]!='c' and dirs[-1]!='t':
#        ID = int(dirs)
#        for q,pr in enumerate(pre):
#            completeoverlap[q]+=data[ID][q]

#for pr,q,ax,lvl in zip(pre,range(len(axs)),axs,levels):
#    print(type(pr))
#    ax.contour(np.arange(-10,10.5,0.5),np.arange(-10,10.5,0.5),completeoverlap[q]/lens[q],colors='k',linewidths=1,levels=lvl)
#    ax.scatter(0,0,marker='o',color='r')
#
#    ax.text(0.05,0.9,'%s hPa'%pr,fontsize=6,transform=ax.transAxes)
#
#fig.savefig(pi + 'streamer-overlap.png',dpi=300,bbox_inches='tight')
#plt.close('all')




    
