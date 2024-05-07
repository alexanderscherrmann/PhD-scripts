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
parser.add_argument('deltaPSP',default='',type=int,help='difference between surface pressure and pressure that should be evaluated as orographical influence')

parser.add_argument('ZBB',default='',type=int,help='evelation in m at which PV changes should be evaluated as orographic')

args = parser.parse_args()
rdis = int(args.rdis)
deltaPSP = int(args.deltaPSP)
zbb = int(args.ZBB)

deltaLONLAT = helper.convert_radial_distance_to_lon_lat_dis(rdis)

pload = '/home/ascherrmann/009-ERA-5/MED/ctraj/use/'
pload2 = '/home/ascherrmann/009-ERA-5/MED/traj/use/'
LON=np.round(np.linspace(-180,180,721),1)
LAT=np.round(np.linspace(-90,90,361),1)

savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']
var = []

for u,x in enumerate(savings):
    f = open(pload2[:-4] + x + 'furthersel.txt',"rb")
    var.append(pickle.load(f))
    f.close()

Slp = var[0]
SLP = var[0]
Clon = var[1]
Clat = var[2]
ID = var[3]
hourstoSLPmin = var[4]
HourstoSLPmin = var[4]
dates = var[5]


avaID = np.array([])
maturedates = np.array([])
SLPminid = np.array([])
end = np.array([])

for k in range(len(ID)):
    avaID=np.append(avaID,ID[k][0].astype(int))
    loc = np.where(hourstoSLPmin[k]==0)[0][0]
    maturedates = np.append(maturedates,dates[k][loc])
    SLPminid = np.append(SLPminid,loc)
    end = np.append(end,maturedates[k] + '-ID-%06d.txt'%avaID[k])


traj = np.array([])

for d in os.listdir(pload):
    if(d.startswith('trajectories-mature')):
            traj = np.append(traj,d)

fsl=6

labs = helper.traced_vars_ERA5MED()
linestyle = ['-',':']
### INFO
### Trajetories start 200km around the center between 975 and 700 hPa
###

wql = 0
meandi = dict()
oro = 'oro'
env = 'env'
cyc = 'cyc'
split=[cyc,env]
pllegend = split
ORO = dict()
datadi = dict() ####raw data of traced vars
dipv = dict() ####splited pv is stored here
dit = dict() ### 1 and 0s, 1 if traj is close to center at given time, 0 if not
meandi[env] = dict()
meandi[cyc] = dict()

H = 48
xlim = np.array([-1*H,0])
#total
ylim = np.array([-0.3,1.5])
#inside pressure layers 
pressure_stack = np.zeros(H+1)
SLPminid = SLPminid.astype(int)
idsave = np.array([])
datesave = np.array([])
highORO = np.array([])

#test = np.where(traj=='trajectories-mature-20200309_15-ID-544480.txt')[0][0]

for uyt2, txt in enumerate(traj[:]):

    cycID = txt[-10:-4]
    idsave = np.append(idsave,cycID)
    date=txt[-25:-14]
    lfd = txt[-25:]
    uyt = np.where(end==lfd)[0][0]

    datesave = np.append(datesave,date)

    datadi[cycID]=dict() #raw data
    dipv[cycID]=dict()    #accumulated pv is saved here
    dipv[cycID][env]=dict()
    dipv[cycID][cyc]=dict()
    
    ORO[cycID]=dict()
    ORO[cycID][cyc]=dict()
    ORO[cycID][env]=dict()

    tt = np.loadtxt(pload + txt)
    for k, el in enumerate(labs):
        datadi[cycID][el] = tt[:,k].reshape(-1,H+1)

    OL = datadi[cycID]['OL']
    ZB = datadi[cycID]['ZB']
    dp = datadi[cycID]['PS'] - datadi[cycID]['P']

    datadi[cycID]['highORO'] = 0

    tmpclon= np.array([])
    tmpclat= np.array([])

    ### follow cyclone backwards to find its center
    dit[cycID] = dict()
    dit[cycID][env] = np.zeros(datadi[cycID]['time'].shape)
    dit[cycID][cyc] = np.zeros(datadi[cycID]['time'].shape)
    dit[cycID][oro] = np.zeros(datadi[cycID]['time'].shape)
    
    for k in range(0,H+1):
        if np.any(hourstoSLPmin[uyt]==(-1*k)):
            tmpclon = np.append(tmpclon,Clon[uyt][np.where(hourstoSLPmin[uyt]==(-k))[0][0]])
            tmpclat = np.append(tmpclat,Clat[uyt][np.where(hourstoSLPmin[uyt]==(-k))[0][0]])
            
        else:
            ### use boundary position that no traj should be near it
            tmpclon = np.append(tmpclon,170)
            tmpclat = np.append(tmpclat,-90)

    ### check every hours every trajecttory whether it is close to the center ###
    for e, h in enumerate(datadi[cycID]['time'][0,:]):
        ### center lon and latitude
        CLON = tmpclon[e]
        CLAT = tmpclat[e]

        ### 30.10.2020 radial distance instead of square
        ### if traj is in circle of 200km set cyc entry to 0, else env entry 1
        for tr in range(len(datadi[cycID]['time'])):
            if (helper.convert_dlon_dlat_to_radial_dis_new(CLON-datadi[cycID]['lon'][tr,e],CLAT-datadi[cycID]['lat'][tr,e],CLAT)<=rdis):
#            if ( np.sqrt( (CLON-datadi[cycID]['lon'][tr,e])**2 + (CLAT-datadi[cycID]['lat'][tr,e])**2) <=  deltaLONLAT):
            ###
                dit[cycID][cyc][tr,e]=1
            else:
                dit[cycID][env][tr,e]=1

            if ((OL[tr,e]>0.7)& (ZB[tr,e]>zbb) & (dp[tr,e]<deltaPSP)):
                dit[cycID][oro][tr,e]=1



    deltaPV = np.zeros(datadi[cycID]['time'].shape)
    deltaPV[:,1:] = datadi[cycID]['PV'][:,:-1]-datadi[cycID]['PV'][:,1:]
    ttmp = dict()
    for key in split:
        ttmp[key] = (dit[cycID][key][:,:-1]+dit[cycID][key][:,1:])/2.
        ttmp[oro] = (dit[cycID][oro][:,:-1]+dit[cycID][oro][:,1:])/2.
        dipv[cycID][key] = np.zeros(datadi[cycID]['time'].shape)
        dipv[cycID][key][:,:-1] = np.flip(np.cumsum(np.flip(deltaPV[:,1:]*ttmp[key][:,:],axis=1),axis=1),axis=1)
        #dipv[cycID][key][:,:-1] = np.flip(np.cumsum(np.flip(deltaPV[:,1:]*dit[cycID][key][:,1:],axis=1),axis=1),axis=1)
        ORO[cycID][key] = np.zeros(datadi[cycID]['time'].shape)
        ORO[cycID][key][:,:-1] = np.flip(np.cumsum(np.flip(deltaPV[:,1:]*ttmp[key][:,:]*ttmp[oro][:,:],axis=1),axis=1),axis=1)
        #ORO[cycID][key][:,:-1] = np.flip(np.cumsum(np.flip(deltaPV[:,1:]*dit[cycID][key][:,1:]*dit[cycID][oro][:,1:],axis=1),axis=1),axis=1)


    ### PLOTTING
    idp = np.where(datadi[cycID]['PV'][:,0]>=0.75)[0]

    if (len(idp)!=0):
     calpre = np.sum(ORO[cycID][env][idp,:],axis=0)/len(idp)
     if (calpre[0]!=0):

      if ((calpre[0]>0.3) & ((calpre[12]/calpre[0]) > 0.75) & (np.mean(dipv[cycID]['env'][:,0])>0.2)):
       if(hourstoSLPmin[uyt][0]<-5):

        highORO = np.append(highORO,cycID)
#        fig, axes = plt.subplots(2,1,sharex=True, sharey=True)
#        plt.subplots_adjust(left=0.15,bottom=None,top=None,right=None,hspace=0.2,wspace=0.1)
#
#        axes = axes.flatten()
#        gax = fig.add_subplot(111, frameon=False)
#        gax.set_xticks(ticks=[])
#        gax.set_yticks(ticks=[])
#    
#        t = datadi[cycID]['time'][0]
#        for q, ax in enumerate(axes):
#                    ax.plot([],[],marker=None,ls='-',color='black')
#                    ax.plot([],[],marker=None,ls=':',color='black')
#    
#                    sq=np.sum(dit[cycID][cyc][idp,:] + dit[cycID][env][idp,:],axis=0)
#                    if(q==0):
#                        tmpdata = dipv[cycID]
#                    else:
#                        tmpdata = ORO[cycID]
#    
#                    for pl,key in enumerate(split):
#                        meantmp = np.array([])
#                        stdtmp = np.array([])
#                        for xx in range(len(sq)):
#                            if sq[xx]>0:
#                                meantmp = np.append(meantmp,np.sum(tmpdata[key][idp,xx])/sq[xx])
#                            else:
#                                meantmp = np.append(meantmp,0)
#                        ax.plot(t,meantmp,color='k',ls=linestyle[pl])
#                    ax.axvline(hourstoSLPmin[uyt][0],color='grey',ls='-')
#    
#                    ax.set_xlim(xlim)
#                    ax.set_ylim(ylim)
#                    ax.set_xticks(ticks=np.arange(-48,1,6))
#                    ax.tick_params(labelright=False,right=True)        
#    
#        fig.text(0.5,0.94, dates[uyt][-1] + ' ' +  str(cycID) ,ha='center',fontsize=10)
#    
#        axes[0].set_ylabel('acc. PV [PVU]')
#        axes[1].set_ylabel('orographic PV [PVU]')
#    
#        axes[1].legend(pllegend,fontsize=fsl,loc='upper left')
#        axes[1].set_xlabel('time until mature stage [h]')
#    
#        name = 'PVsplit-' + date +  '-' + '%06d-'%int(cycID) + 'PSP-' + str(deltaPSP) + '-ZB-' + str(zbb) +'-' +  str(rdis) + '.png'
#        fig.savefig(pload[:-4] + '/fig/' + name,dpi=300,bbox_inches="tight")
#        plt.close()

    wql=10
    pressure_stack = np.vstack((pressure_stack,datadi[cycID]['P']))

PVdata = dict()
PVdata['dipv'] = dipv
PVdata['oro'] = ORO
PVdata['ids'] = idsave
PVdata['rawdata'] = datadi
PVdata['dates'] = datesave
PVdata['highORO'] = highORO
PVdata['dit'] = dit

f = open(pload + 'PV-data-' + 'dPSP-' + str(deltaPSP) + '-ZB-' + str(zbb) + '-2-%d-correct-distance.txt'%rdis,'wb')
pickle.dump(PVdata,f)
f.close()
