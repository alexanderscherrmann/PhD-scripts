import numpy as np
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm
import argparse
import pickle

parser = argparse.ArgumentParser(description="plot accumulated average PV gain that is associated with the cyclone and the environment")
parser.add_argument('rdis',default='',type=int,help='distance from center in km that should be considered as cyclonic')
args = parser.parse_args()

p = '/home/ascherrmann/009-ERA-5/'

traced = np.array([])
for d in os.listdir(p):
    if(d.startswith('traced-vars-20')):
            traced = np.append(traced,d)
traced = np.sort(traced)

fsl=4

f = open(p + 'lats-Manos.txt','rb')
clat = pickle.load(f)
f.close()

f = open(p + 'lons-Manos.txt','rb')
clon = pickle.load(f)
f.close()

f = open(p + 'dates-Manos.txt','rb')
dates = pickle.load(f)
f.close()

labs = helper.traced_vars_ERA5()

ptitle = np.array(['700 hPa < P$_{0}$', 'P$_{0} <$ 700 hPa', '400 < P$_{-48} <$ 600 hPa',  'P$_{-48} <$ 400 hPa'])
linestyle = ['-',':']

LON=np.linspace(-120,30,301)
LAT=np.linspace(-20,90,221)

rdis = int(args.rdis)
deltaLONLAT = helper.convert_radial_distance_to_lon_lat_dis(rdis)

### INFO
### Trajetories start 200km around the center between 925 and 600 hPa if PV>0.6 PVU
###

wql = 0
meandi = dict()
env = 'env'
cyc = 'cyc'
split=[cyc,env]

datadi = dict() ####raw data of traced vars
dipv = dict() ####splited pv is stored here
dit = dict() ### 1 and 0s, 1 if traj is close to center at given time, 0 if not
meandi[env] = dict()
meandi[cyc] = dict()

H = 47
xlim = np.array([-1*H,0])
#total
ylim = np.array([-0.3,1.0])
#inside pressure layers 
ylim2 = np.array([-0.3,0.5])
#xlim = xlim - 12
pressure_stack = np.zeros(H+1)
#IDstart = np.append([0],np.where(htzeta[1:]<htzeta[:-1])[0]+1)

#hoursegments = np.flip(np.arange(-48,1,1))
linewidth=1.5
alpha=1.
#cmap = ListedColormap(['saddlebrown','orange'])
#norm = BoundaryNorm([0, 0.5, 1], cmap.N)

for uyt, txt in enumerate(traced[:]):
    cycID=txt[-10:-4]
    date=txt[-25:-14]
    yyyy = int(date[0:4])
    MM = int(date[4:6])
    DD = int(date[6:8])
    hh = int(date[9:])

    datadi[cycID]=dict() #raw data
    dipv[cycID]=dict()    #accumulated pv is saved here
    dipv[cycID][env]=dict()
    dipv[cycID][cyc]=dict()

    tt = np.loadtxt(p + txt)
    for k, el in enumerate(labs):
        datadi[cycID][el] = tt[:,k].reshape(-1,H+1)

    tmpclon= np.array([])
    tmpclat= np.array([])

    ### follow cyclone backwards to find its center
    dit[cycID] = dict()
    dit[cycID][env] = np.zeros(datadi[cycID]['time'].shape)
    dit[cycID][cyc] = np.zeros(datadi[cycID]['time'].shape)
    alpha=0
    for k in range(0,H+1):
        if (hh<0):
            hh=23
            DD-=1
            if(DD<1):
                MM-=1
                DD=helper.month_days(yyyy)[int(MM)-1]

        Date=str(yyyy)+'%02d%02d_%02d'%(MM,DD,hh)


        if(np.where((dates[uyt]==Date))[0].size):
            alpha+=1
            tmpq = np.where((dates[uyt]==Date))[0][0]
            hh-=1
            tmpclon = np.append(tmpclon,np.where(np.round(LON,2)==np.round(clon[uyt][tmpq],2))[0].astype(int))
            tmpclat = np.append(tmpclat,np.where(np.round(LAT,2)==np.round(clat[uyt][tmpq],2))[0].astype(int))
            if (np.where(np.round(LAT,2)==np.round(clat[uyt][tmpq],2))[0].size==0):
                tmpclat = np.append(tmpclat,np.where(np.round(LAT,2)==np.round(clat[uyt][tmpq],0))[0].astype(int))
            if (np.where(np.round(LON,2)==np.round(clon[uyt][tmpq],2))[0].size==0):
                tmpclon = np.append(tmpclon,np.where(np.round(LON,2)==np.round(clon[uyt][tmpq],0))[0].astype(int))
        else:
            ### use boundary position that no traj should be near it
            tmpclon = np.append(tmpclon,300)
            tmpclat = np.append(tmpclat,0)
            alpha+=1
    ### check every hours every trajecttory whether it is close to the center ###
    for e, h in enumerate(datadi[cycID]['time'][0,:]):
        tmplon = tmpclon[e].astype(int)
        tmplat = tmpclat[e].astype(int)
        ### center lon and latitude
        CLON = LON[tmplon]
        CLAT = LAT[tmplat]

        ### 30.10.2020 radial distance instead of square
        ### if traj is in circle of 200km set cyc entry to 0, else env entry 1
        for tr in range(len(datadi[cycID]['time'])):
            if ( np.sqrt( (CLON-datadi[cycID]['lon'][tr,e])**2 + (CLAT-datadi[cycID]['lat'][tr,e])**2) <=  deltaLONLAT):
            ###
                dit[cycID][cyc][tr,e]=1
            else:
                dit[cycID][env][tr,e]=1

    fig, axes = plt.subplots(1,2,sharex=True, sharey=True)
    plt.subplots_adjust(left=0.15,bottom=None,top=None,right=None,hspace=0.2,wspace=0.1)
    axes = axes.flatten()
    gax = fig.add_subplot(111, frameon=False)
    gax.set_xticks(ticks=[])
    gax.set_yticks(ticks=[])

    deltaPV = np.zeros(datadi[cycID]['time'].shape)
    deltaPV[:,1:-1] = datadi[cycID]['PV'][:,1:-1]-datadi[cycID]['PV'][:,2:]

    for key in split:

        dipv[cycID][key] = np.zeros(datadi[cycID]['time'].shape)
        dipv[cycID][key][:,:-2] = np.flip(np.cumsum(np.flip(deltaPV[:,1:-1]*dit[cycID][key][:,1:-1],axis=1),axis=1),axis=1)

        if wql==0:
            meandi[key] = dipv[cycID][key]
        else:
            meandi[key] = np.concatenate((meandi[key], dipv[cycID][key]),axis=0)
    ### PLOTTING
    t = datadi[cycID]['time'][0]
    for q, ax in enumerate(axes):
                ax.plot([],[],marker=None,ls='-',color='black')
                ax.plot([],[],marker=None,ls=':',color='black')
                if q==0:
                    idp = np.where(datadi[cycID]['P'][:,0]>700)[0]
                #    print(cycID, len(np.where((dipv[cycID][cyc]['APVTOT'][idp,0] + dipv[cycID][env]['APVTOT'][idp,0])>datadi[cycID]['PV'][idp,0])[0])/len(datadi[cycID]['time'][idp,0]))
                else:
                    idp = np.where(datadi[cycID]['P'][:,0]<700)[0]
                ### this is for all traj, also possible to weight the mean by the traj number in that 
                ### pressure regime
#                sq=np.sum(dit[cycID][cyc][idp] + dit[date][env][idp],axis=0)
                sq=np.sum(dit[cycID][cyc] + dit[cycID][env],axis=0)
                for pl,key in enumerate(split):
                    meantmp = np.array([])
                    stdtmp = np.array([])
                    for xx in range(len(sq)):
                        if sq[xx]>0:
                            meantmp = np.append(meantmp,np.sum(dipv[cycID][key][idp,xx])/sq[xx])
                            stdtmp = np.append(stdtmp,np.sqrt(np.sum( (dipv[cycID][key][idp,xx]-meantmp[-1])**2 )/sq[xx]))
                        else:
                            meantmp = np.append(meantmp,0)
                            stdtmp = np.append(stdtmp,np.sqrt(np.sum((dipv[cycID][key][idp,xx]-meantmp[-1])**2 )/1))
                    ax.plot(t,meantmp,color='k',ls=linestyle[pl])
                ax.axvline(-1 * len(dates[uyt])+1,color='grey',ls='-')

                ax.set_xlim(xlim)
                ax.set_ylim(ylim2)
                ax.set_title(ptitle[q],fontsize=8)
                ax.set_xticks(ticks=np.arange(-48,1,6))
                ax.tick_params(labelright=False,right=True)        

    fig.text(0.5,0.94, dates[uyt][-1] + ' ' +  str(cycID) ,ha='center',fontsize=10)
    fig.text(0.5, 0.04, 'time until mature stage [h]',ha='center')
    fig.text(0.04,0.5, 'accumulated PV [PVU]',va='center',rotation='vertical')
    name = 'PVchange-' + date +  '-' + str(cycID) + '-endpressure-' + str(rdis) + '.png'
    fig.savefig(p + name,dpi=300,bbox_inches="tight")
    plt.close()

    wql=10
    pressure_stack = np.vstack((pressure_stack,datadi[cycID]['P']))

pressure_stack=np.delete(pressure_stack,0,axis=0)

ditstack = dict()
for key in split:
    for k,txt in enumerate(traced[:]):
        cycID=txt[-10:-4]
        
        if k==0:
            ditstack[key] = dit[cycID][key]
        else:
            ditstack[key] = np.concatenate((ditstack[key],dit[cycID][key]),axis=0)



fig, axes = plt.subplots(1,2,sharex=True, sharey=True)
plt.subplots_adjust(left=0.15,bottom=None,top=None,right=None,hspace=0.2,wspace=0.1)
axes = axes.flatten()
gax = fig.add_subplot(111, frameon=False)
gax.set_xticks(ticks=[])
gax.set_yticks(ticks=[])

titles = 'composite-' + 'PV-contribution'
for pl, key in enumerate(split):
    for q, ax in enumerate(axes):
        ax.plot([],[],marker=None,ls='-',color='black')
        ax.plot([],[],marker=None,ls=':',color='black')
        if q==0:
            idp = np.where(pressure_stack[:,0]>700)[0]
        else:
                idp = np.where(pressure_stack[:,0]<700)[0]

        sq=np.sum(ditstack[env]+ ditstack[cyc],axis=0)
        meantmp = np.array([])
        stdtmp = np.array([])

        for xx in range(len(sq)):
            if sq[xx]>0:
                meantmp = np.append(meantmp,np.sum(meandi[key][idp,xx])/sq[xx])
                stdtmp = np.append(stdtmp,np.sqrt(np.sum( (meandi[key][idp,xx]-meantmp[-1])**2 )/sq[xx]))
            else:
                meantmp = np.append(meantmp,0)
                stdtmp = np.append(stdtmp,np.sqrt(np.sum((meandi[key][idp,xx]-meantmp[-1])**2 )/1))
        
        ax.plot(t,meantmp,color='k',ls=linestyle[pl])

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axvline(-1 * len(dates[uyt])+1,color='grey',ls='-')
        ax.set_title(ptitle[q],fontsize=8)
        ax.set_xticks(ticks=np.arange(-48,1,6))
        ax.tick_params(labelright=False,right=True)

fig.text(0.5, 0.04, 'time until mature stage [h]',ha='center')
fig.text(0.04,0.5, 'accumulated PV [PVU]',va='center',rotation='vertical')

name = 'PVchange-composite-endpressure-' + str(rdis) +  '.png'
fig.savefig(p + name,dpi=300,bbox_inches="tight")
plt.close()

