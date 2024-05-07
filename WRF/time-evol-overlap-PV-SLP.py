from netCDF4 import Dataset as ds
import numpy as np
import os
import pandas as pd
import pickle 

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib.colors import BoundaryNorm

import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper


minlon = -10
minlat = 25
maxlat = 50
maxlon = 45

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
cycloneSLP = '/atmosdyn/michaesp/mincl.era-5/tracks/'

df = pd.read_csv('/atmosdyn2/ascherrmann/011-all-ERA5/data/pandas-basic-data-all-deep-over-sea-12h.csv')
trange=np.arange(-168,49,3)

dID = df['ID'].values
mdates = df['dates'].values
dID = df['ID'].values
counter = np.zeros(5)
for q,pr in enumerate(['250','300','350','400','450']):
    fig,axes = plt.subplots(2,1,sharex=True)
    axes = axes.flatten()
    gig,gaxes = plt.subplots(2,1,sharex=True)
    gaxes=gaxes.flatten()
    sca,sax = plt.subplots()
    sca2,sax2 = plt.subplots()
    for dirs in os.listdir(ps):
        if dirs[-1]!='c' and dirs[-1]!='t':
            ID = dirs +'/'

            loc = np.where(dID==int(ID[:-1]))[0][0]
            yyyy = mdates[loc][:4]
            mm = mdates[loc][4:6]
            slptrack = np.loadtxt(cycloneSLP + 'fi_' + yyyy + mm,skiprows=4)
            if not np.any(slptrack[:,-1]==int(dirs)):
                mm = int(mm)-1
                if mm<1:
                    mm=12
                    yyyy = '%d'%(int(yyyy)-1)

                slptrack = np.loadtxt(cycloneSLP + 'fi_' + yyyy + '%02d'%mm,skiprows=4)

            cycslptrack = slptrack[np.where(slptrack[:,-1]==int(dirs))[0],3]
            minslp = np.where(cycslptrack==np.min(cycslptrack))[0][0]

            data = np.loadtxt(ps + ID + 'overlapping-streamer-tracks-%s.txt'%pr,skiprows=1)
            if data.size==0:
                continue

            data = data.reshape(-1,10)
            t = data[:,0]
            ut = np.unique(t)
            slp = data[:,-1]
            ovintPV = data[:,-2]
            ovarea = data[:,2]
            slpplt = np.array([])
            tplt = np.array([])
            maxovPV = np.array([])
            for u in ut:
                ul = np.where(t==u)[0]
                if len(ul)==1:
                    if slp[ul]==0:
                        continue
                    tplt = np.append(tplt,u)
                    maxovPV =np.append(maxovPV,ovintPV[ul[0]])
                    slpplt = np.append(slpplt,slp[ul[0]])
                else:
                    slpplt = np.append(slpplt,slp[ul[np.argmax(ovarea[ul])]])
                    tplt = np.append(tplt,t[ul[np.argmax(ovarea[ul])]])
                    maxovPV=np.append(maxovPV,ovintPV[ul[np.argmax(ovarea[ul])]])
           
            
            if np.any(tplt==0):
                tl = np.where(tplt==0)[0][0]
                pvl = np.where(maxovPV==np.max(maxovPV[:tl+1]))[0][0]

                dslp = slp[tl] - slp[pvl]
                dt = t[pvl]
                dpv = ovintPV[tl] - ovintPV[pvl]
                sax.scatter(dslp,dpv,color='k')
                sax2.scatter(dt,dslp,color='k')
                counter[q]+=1
            else:
                try:
                    tl = np.where(tplt<0)[0][-1]
                except:
                    continue
                
                maxpv = np.where(maxovPV[:tl+1]==np.max(maxovPV[:tl+1]))[0][0]
                dt = t[maxpv]
                dslp = cycslptrack[minslp]-slp[tl]
                dpv = -ovintPV[maxpv]
                sax.scatter(dslp,dpv,color='k')
                sax2.scatter(dt,dslp,color='k')
                tl = 0
                counter[q]+=1
                



            pvl = np.where(maxovPV==np.max(maxovPV))[0][0]

            if pvl < tl:
                ax = gaxes[0]
            else:
                ax = axes[0]

            ax.plot(tplt,maxovPV,color='k')

            if pvl < tl:
                ax =gaxes[1]
            else:
                ax = axes[1]
            ax.plot(tplt,slpplt,color='k')

            
    
    axes[1].set_xlabel('time to mature stage [h]')
    axes[1].set_ylabel('SLP [hPa]')
    axes[0].set_ylabel(r'$\int_{\mathrm{overlap}} \mathrm{PV}$ [PVU]')

    gaxes[1].set_xlabel('time to mature stage [h]')
    gaxes[1].set_ylabel('SLP [hPa]')
    gaxes[0].set_ylabel(r'$\int_{\mathrm{overlap}} \mathrm{PV}$ [PVU]')
    sax.set_xlabel('SLP [hPa]')
    sax.set_ylabel(r'$\int_{\mathrm{overlap}} \mathrm{PV}$ [PVU]')
    sax2.set_xlim(-24,0)
    sax2.set_xlabel(r'$\Delta\mathrm{t}_{\mathrm{minSLP,maxPV}}$ [h]')
    sax2.set_ylabel(r'$\Delta$SLP [hPa]')
    #plt.adjust_subplot(wspace=0,hspace=0)
    fig.savefig(pi + 'overlap-PV-SLP-time-evolution-at-%s.png'%pr,dpi=300,bbox_inches='tight')
    gig.savefig(pi + 'prior-PVmax-overlap-PV-SLP-time-evo-at-%s.png'%pr,dpi=300,bbox_inches='tight')
    sca.savefig(pi + 'scatter-PV-slp-mature-%s.png'%pr,dpi=300,bbox_inches='tight')
    sca2.savefig(pi + 'scatter-delta-t-delta-SLP-%s.png'%pr,dpi=300,bbox_inches='tight')
    plt.close('all')

print(counter)
    
