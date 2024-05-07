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
import pickle
import argparse

parser = argparse.ArgumentParser(description="plot accumulated average PV gain that is associated with the cyclone and the environment")
parser.add_argument('rdis',default='',type=int,help='distance from center in km that should be considered as cyclonic')
parser.add_argument('type',default='',type=str,help='MED, TRO or ETA')

parser.add_argument('deltaPSP',default='',type=int,help='difference between surface pressure and pressure that should be evaluated as orographical influence')

parser.add_argument('ZBB',default='',type=int,help='evelation in m at which PV changes should be evaluated as orographic')

args = parser.parse_args()
CT = str(args.type)
rdis = int(args.rdis)
deltaPSP = int(args.deltaPSP)
zbb = int(args.ZBB)

pload = '/home/ascherrmann/TT/use/' 
plload = '/home/ascherrmann/010-IFS/traj/' + CT + '/'

traj = np.array([])
for d in os.listdir(pload):
    if(d.startswith('trajectories-mature-')):
            traj = np.append(traj,d)            

MON = np.array([])
for d in os.listdir(plload):
    if(d.startswith('traend-')):
        MON = np.append(MON,d)

MON = np.sort(MON)
traj = np.sort(traj)

mon = np.array([])
for k in traj:
    for l in MON:
        if k[-25:-4]==l[-31:-10]:
            mon = np.append(mon,l)
MON=mon


fsl=6

labs = helper.traced_vars_IFS()

cl=['k','orange','green','dodgerblue','blue','red']
pllegend = ['cyc','env','TOT','|CONVT|>|TURBT|','|TURBT|>|CONVT|', 'CONVM', 'TURBM','RAD','LS']


plotvars = ['APVTOT','PVR-T','PVRCONVM','PVRTURBM','APVRAD','PVRLS']
linestyle = ['-',':']

LON=np.linspace(-180,180,901)
LAT=np.linspace(0,90,226)
deltaLONLAT = helper.convert_radial_distance_to_lon_lat_dis(rdis)

f = open('/home/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','rb')
td = pickle.load(f)
f.close()

wql = 0
meandi = dict()
env = 'env'
cyc = 'cyc'
oro = 'oro'
ORO = dict()
split=[cyc,env]

datadi = dict() ####raw data of traced vars
dipv = dict() ####splited pv is stored here
dit = dict() ### 1 and 0s, 1 if traj is close to center at given time, 0 if not
meandi[env] = dict()
meandi[cyc] = dict()

H = 96
xlim = np.array([-1*H,0])
#total
ylim = np.array([-0.3,1.25])

pressure_stack = np.zeros(H+1)

pvsum = np.where(labs=='PVRCONVT')[0][0]

hoursegments = np.flip(np.arange(-1*H,1,1))
linewidth=1.5
alpha=1.
cmap = ListedColormap(['saddlebrown','orange'])
norm = BoundaryNorm([0, 0.5, 1], cmap.N)

monsave = np.array([])
idsave = np.array([])
datesave = np.array([])
highORO = np.array([])
for uyt, txt in enumerate(traj):

    montmp = MON[uyt][-9:-4]
    monsave = np.append(monsave,montmp)

    idtmp = int(txt[-10:-4])
    idsave = np.append(idsave,idtmp)


    date=txt[-25:-14]
    if date!='20171223_15':
        continue

    date=date+'-%03d'%idtmp
    datesave=np.append(datesave,date)

    datadi[date]=dict() #raw data

    dipv[date]=dict()    #accumulated pv is saved here
    dipv[date][env]=dict()
    dipv[date][cyc]=dict()

    ORO[date] = dict()
    ORO[date][env]=dict()
    ORO[date][cyc]=dict()

    htzeta = td[montmp][idtmp]['hzeta']
    zeroid = np.where(htzeta==0)[0][0]
    htzeta = htzeta[:zeroid+1]
    clat = td[montmp][idtmp]['clat'][:zeroid+1]
    clon = td[montmp][idtmp]['clon'][:zeroid+1]
    

    tt = np.loadtxt(pload + txt)
    for k, el in enumerate(labs):
        datadi[date][el] = tt[:,k].reshape(-1,H+1)

    dp = datadi[date]['PS']-datadi[date]['P']
    OL = datadi[date]['OL']
    ZB = datadi[date]['ZB']

    tmpclon= np.array([])
    tmpclat= np.array([])

    ### follow cyclone backwards to find its center
    dit[date] = dict()
    dit[date][env] = np.zeros(datadi[date]['time'].shape)
    dit[date][cyc] = np.zeros(datadi[date]['time'].shape)
    dit[date][oro] = np.zeros(datadi[date]['time'].shape)

    for k in range(0,H+1):
        if(np.where(htzeta==(-k))[0].size):
            tmpq = np.where(htzeta==(-k))[0][0]
            tmpclon = np.append(tmpclon,np.mean(clon[tmpq]))
            tmpclat = np.append(tmpclat,np.mean(clat[tmpq]))
        else:
            ### use boundary position that no traj should be near it
            tmpclon = np.append(tmpclon,860)
            tmpclat = np.append(tmpclat,0)
    deltaPV = np.zeros(datadi[date]['PV'].shape)
    deltaPV[:,1:] = datadi[date]['PV'][:,:-1]-datadi[date]['PV'][:,1:]
    ### check every hours every trajecttory whether it is close to the center ###
    for e, h in enumerate(datadi[date]['time'][0,:]):

        tmplon = tmpclon[e].astype(int)
        tmplat = tmpclat[e].astype(int)

        ### center lon and latitude
        CLON = LON[tmplon]
        CLAT = LAT[tmplat]

        ### 30.10.2020 radial distance instead of square
        ### if traj is in circle of 200km set cyc entry to 0, else env entry 1
        for tr in range(len(datadi[date]['time'])):
            if ( np.sqrt( (CLON-datadi[date]['lon'][tr,e])**2 + (CLAT-datadi[date]['lat'][tr,e])**2) <=  deltaLONLAT):
            ###
                dit[date][cyc][tr,e]=1
            else:
                dit[date][env][tr,e]=1

            ### check for orography
            if ((OL[tr,e]>0.7) & (ZB[tr,e]>zbb) & (dp[tr,e]<deltaPSP)):
                dit[date][oro][tr,e] = 1


    for key in split:
        for k, el in enumerate(labs[pvsum:]):
            dipv[date][key][el] = np.zeros(datadi[date]['time'].shape)
            dipv[date][key][el][:,:-1] = np.flip(np.cumsum(np.flip((datadi[date][el][:,1:] + datadi[date][el][:,:-1])/2.*dit[date][key][:,1:],axis=1),axis=1),axis=1)

            ORO[date][key][el] = np.zeros(datadi[date]['time'].shape)
            ORO[date][key][el][:,:-1] = np.flip(np.cumsum(np.flip((datadi[date][el][:,1:] + datadi[date][el][:,:-1])/2.*dit[date][key][:,1:] * dit[date][oro][:,1:],axis=1),axis=1),axis=1)
        
        dipv[date][key]['deltaPV'] = np.zeros(datadi[date]['time'].shape)
        dipv[date][key]['deltaPV'][:,:-1] = np.flip(np.cumsum(np.flip(deltaPV[:,1:] * dit[date][key][:,1:],axis=1),axis=1),axis=1)

        dipv[date][key]['APVTOT'] =np.zeros(datadi[date]['time'].shape)
        ORO[date][key]['APVTOT'] = np.zeros(datadi[date]['time'].shape)
        for el in labs[pvsum:]:
            dipv[date][key]['APVTOT'] += dipv[date][key][el]
            ORO[date][key]['APVTOT'] += ORO[date][key][el]

        dipv[date][key]['APVRAD'] = dipv[date][key]['PVRSW'] + dipv[date][key]['PVRLWH'] + dipv[date][key]['PVRLWC']
        dipv[date][key]['PVR-T'] = dipv[date][key]['PVRTURBT'] + dipv[date][key]['PVRCONVT']

        ORO[date][key]['APVRAD'] = ORO[date][key]['PVRSW'] + ORO[date][key]['PVRLWH'] + ORO[date][key]['PVRLWC']
        ORO[date][key]['PVR-T'] = ORO[date][key]['PVRTURBT'] + ORO[date][key]['PVRCONVT']

#        for el in np.append(labs[pvsum:],['APVTOT','APVRAD','PVR-T']):
#            if wql==0:
#                meandi[key][el] = dipv[date][key][el]
#            else:
#                meandi[key][el] = np.concatenate((meandi[key][el], dipv[date][key][el]),axis=0)


    ### PLOTTING
    #select data
    idp = np.where(datadi[date]['PV'][:,0]>=0.75)[0]
    if (len(idp)!=0):

     calpre = np.sum(ORO[date][env]['APVTOT'][idp,:],axis=0)/len(idp)
     if True:
#     if (calpre[0]!=0):
#      if ((calpre[0]>0.3) & ((calpre[12]/calpre[0]) > 0.85)):
        highORO = np.append(highORO,date)
        fig, axes = plt.subplots(2,1,sharex=True, sharey=True)
        plt.subplots_adjust(left=0.15,bottom=None,top=None,right=None,hspace=0.2,wspace=0.1)
    
        axes = axes.flatten()
    
        gax = fig.add_subplot(111, frameon=False)
        gax.set_xticks(ticks=[])
        gax.set_yticks(ticks=[])
    
        t = datadi[date]['time'][0]
        titles= 'PV-contribution'
        for q, ax in enumerate(axes):
                ax.plot([],[],marker=None,ls='-',color='black')
                ax.plot([],[],marker=None,ls=':',color='black')

                ### this is for all traj, also possible to weight the mean by the traj number in that 
                ### pressure regime

                sq=np.sum(dit[date][cyc][idp,:] + dit[date][env][idp,:],axis=0)
                if(q==0):
                    tmpdata = dipv[date]
                else:
                    tmpdata = ORO[date]

                for pl,key in enumerate(split):
                  for wt, ru in enumerate(plotvars):
                    meantmp = np.array([])
                    stdtmp = np.array([])
                    if ru =='PVR-T':
                        segmentval = np.array([])
                    for xx in range(len(sq)):
                        if sq[xx]>0:
                            meantmp = np.append(meantmp,np.sum(tmpdata[key][ru][idp,xx])/sq[xx])
                            if ru=='PVR-T':
                                if (abs(np.sum(tmpdata[key]['PVRCONVT'][idp,xx]))>=(abs(np.sum(tmpdata[key]['PVRTURBT'][idp,xx])))):
                                    segmentval = np.append(segmentval,1)
                                else:
                                    segmentval = np.append(segmentval,0)
                        else:
                            meantmp = np.append(meantmp,0)
                    if ru =='PVR-T':
                        segments = helper.make_segments(hoursegments,meantmp)
                        lc = mcoll.LineCollection(segments, array=segmentval, cmap=cmap, norm=norm, linestyle=linestyle[pl],linewidth=linewidth, alpha=alpha)
                        ax.add_collection(lc)
                        ax.plot([],[],color='orange',ls='-')
                        ax.plot([],[],color='saddlebrown',ls='-')

                    else:
                        ax.plot(t,meantmp,color=cl[wt],label=pllegend[wt],ls=linestyle[pl])

                ax.axvline(htzeta[0],color='grey',ls='-')
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xticks(ticks=np.arange(-1*H,1,12))
                ax.set_yticks(ticks=np.arange(-0.25,1.26,0.25))
                ax.tick_params(labelright=False,right=True)       

        axes[0].set_ylabel('acc. PV [PVU]')
        axes[1].set_ylabel('orographic PV [PVU]')
    
        axes[1].legend(pllegend,fontsize=fsl,loc='upper left')
        axes[1].set_xlabel('time until mature stage [h]')
        fig.text(0.5,0.94, titles + ' ' + date,ha='center',fontsize=12)
    
        name = '4-day-PV-splits-' +  date + '-PSP-' + str(deltaPSP) + '-ZB-' + str(zbb) +'-' + str(rdis) + '.png'
        fig.savefig(plload + name,dpi=300,bbox_inches="tight")
        plt.close()

    wql=10
    pressure_stack = np.vstack((pressure_stack,datadi[date]['P']))


PVdata = dict()
PVdata['dipv'] = dipv
PVdata['oro'] = ORO
PVdata['mons'] = monsave
PVdata['ids'] = idsave
PVdata['rawdata'] = datadi
PVdata['dates'] = datesave
PVdata['highORO'] = highORO

#f = open(pload + 'PV-data-4days-' + CT + 'dPSP-' + str(deltaPSP) + '-ZB-' + str(zbb) + 'PVedge-0.3.txt','wb')
#pickle.dump(PVdata,f)
#f.close()


