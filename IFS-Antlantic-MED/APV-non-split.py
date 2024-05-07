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

args = parser.parse_args()
CT = str(args.type)
rdis = int(args.rdis)

pload = '/home/ascherrmann/010-IFS/ctraj/' + CT + '/use/' 
plload = '/home/ascherrmann/010-IFS/ctraj/' + CT + '/'

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

fsl=6

labs = helper.traced_vars_IFS()

cl=['k','orange','green','dodgerblue','blue','red']
pllegend = ['apv','TOT','|CONVT|>|TURBT|','|TURBT|>|CONVT|', 'CONVM', 'TURBM','RAD','LS']


plotvars = ['APVTOT','PVR-T','PVRCONVM','PVRTURBM','APVRAD','PVRLS']
linestyle = ['-',':']

LON=np.linspace(-180,180,901)
LAT=np.linspace(0,90,226)

f = open('/home/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','rb')
td = pickle.load(f)
f.close()

### INFO
### Trajetories start 200km around the center between 975 and 700 hPa
###
text = ['a)','b)']
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

H = 48
xlim = np.array([-1*H,0])
#total
ylim = np.array([-0.3,1.25])

pressure_stack = np.zeros(H+1)

pvsum = np.where(labs=='PVRCONVT')[0][0]

hoursegments = np.flip(np.arange(-48,1,1))
linewidth=1.5
alpha=1.
cmap = ListedColormap(['saddlebrown','orange'])
norm = BoundaryNorm([0, 0.5, 1], cmap.N)

monsave = np.array([])
idsave = np.array([])
datesave = np.array([])
for uyt, txt in enumerate(traj):

    montmp = MON[uyt][-9:-4]
    monsave = np.append(monsave,montmp)

    idtmp = int(txt[-10:-4])
    idsave = np.append(idsave,idtmp)

    date=txt[-25:-14]
    if date!='20171214_02' and date!='20180619_03':
#    if date!='20180303_11' and date!='20180203_13':
        continue
#    if date=='20180619_03':
#        montmp='JUN18'
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
    

    tt = np.loadtxt(pload + txt)
    for k, el in enumerate(labs):
        datadi[date][el] = tt[:,k].reshape(-1,H+1)

    deltaPV = np.zeros(datadi[date]['PV'].shape)
    deltaPV[:,1:] = datadi[date]['PV'][:,:-1]-datadi[date]['PV'][:,1:]

    for k, el in enumerate(labs[pvsum:]):
            dipv[date][el] = np.zeros(datadi[date]['time'].shape)
            dipv[date][el][:,:-1] = np.flip(np.cumsum(np.flip((datadi[date][el][:,1:] + datadi[date][el][:,:-1])/2.,axis=1),axis=1),axis=1)

    dipv[date]['deltaPV'] = np.zeros(datadi[date]['time'].shape)
    dipv[date]['deltaPV'][:,:-1] = np.flip(np.cumsum(np.flip(deltaPV[:,1:],axis=1),axis=1),axis=1)

    dipv[date]['APVTOT'] =np.zeros(datadi[date]['time'].shape)
    for el in labs[pvsum:]:
        dipv[date]['APVTOT'] += dipv[date][el]

    dipv[date]['APVRAD'] = dipv[date]['PVRSW'] + dipv[date]['PVRLWH'] + dipv[date]['PVRLWC']
    dipv[date]['PVR-T'] = dipv[date]['PVRTURBT'] + dipv[date]['PVRCONVT']

    ### PLOTTING
    #select data
    idp = np.where(datadi[date]['PV'][:,0]>=0.75)[0]
    if True:
        fig,axes = plt.subplots()
    
        gax = fig.add_subplot(111, frameon=False)
        gax.set_xticks(ticks=[])
        gax.set_yticks(ticks=[])
    
        t = datadi[date]['time'][0]
#        titles= 'PV-contribution'
#        for q, ax in enumerate(axes):
        tmpdata = dipv[date]
        for q in range(0,1):
                ax = axes
                ax.plot(t,np.mean(datadi[date]['PV'][idp],axis=0)-np.mean(datadi[date]['PV'][idp],axis=0)[-1],color='grey')        
                ### this is for all traj, also possible to weight the mean by the traj number in that 
                ### pressure regime
                sq = np.ones(49) * len(idp)
                for wt, ru in enumerate(plotvars):
                    meantmp = np.array([])
                    stdtmp = np.array([])
                    if ru =='PVR-T':
                        segmentval = np.array([])
                    for xx in range(len(sq)):
                        if sq[xx]>0:
                            meantmp = np.append(meantmp,np.sum(tmpdata[ru][idp,xx])/sq[xx])
                            if ru=='PVR-T':
                                if ( abs( np.sum(datadi[date]['PVRCONVT'][idp,xx]) ) >= abs( np.sum( datadi[date]['PVRTURBT'][idp,xx] ) ) ):
                                    segmentval = np.append(segmentval,1)
                                else:
                                    segmentval = np.append(segmentval,0)
                        else:
                            meantmp = np.append(meantmp,0)
                    if ru =='PVR-T':
                        segments = helper.make_segments(hoursegments,meantmp)
                        lc = mcoll.LineCollection(segments, array=segmentval, cmap=cmap, norm=norm, linestyle='-',linewidth=linewidth, alpha=alpha)
                        ax.add_collection(lc)
                        ax.plot([],[],color='orange',ls='-')
                        ax.plot([],[],color='saddlebrown',ls='-')

                    else:
                        ax.plot(t,meantmp,color=cl[wt],label=pllegend[wt],ls='-')

                ax.axvline(htzeta[0],color='grey',ls='-')
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xticks(ticks=np.arange(-48,1,6))
                ax.set_yticks(ticks=np.arange(-0.25,1.26,0.25))
                ax.tick_params(labelright=False,right=True)       

        axes.set_ylabel('acc. PV [PVU]')

#        axes[1].set_ylabel('orographic PV [PVU]')
    
        axes.legend(pllegend,fontsize=fsl,loc='upper left')
        axes.set_xlabel('time until mature stage [h]')
        fig.text(0.5,0.94, date,ha='center',fontsize=12)
        if(date=='20171214_02-073'):
#        if(date=='20180203_13-15'):
            te = text[0]
        else:
            te = text[1]
        fig.text(0.05, 0.95, te,fontsize=12, fontweight='bold', va='top')
        name = 'APV-' +  date + '-' + str(rdis) + '.png'
        fig.savefig(plload + name,dpi=300,bbox_inches="tight")
        plt.close()

