import numpy as np
import xarray as xr
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.stats.stats import pearsonr
import pickle

pload = '/home/ascherrmann/009-ERA-5/MED/traj/use/'
psave = '/home/ascherrmann/009-ERA-5/MED/'

f = open(pload + 'PV-data-dPSP-100-ZB-800-2-400.txt','rb')
PVdata = pickle.load(f)
f.close()

datadi = PVdata['rawdata']
dipv = PVdata['dipv']
ORO = PVdata['oro']

both = np.array([])
adv = np.array([])
cyclonic = np.array([])
environmental = np.array([])

savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']
var = []

for u,x in enumerate(savings):
    f = open(pload[:-4] + x + 'furthersel.txt',"rb")
    var.append(pickle.load(f))
    f.close()

SLP = var[0]
lon = var[1]
lat = var[2]
ID = var[3]
hourstoSLPmin = var[4]
dates = var[5]
avaID = np.array([])
minSLP = np.array([])
for k in range(len(ID)):
    avaID=np.append(avaID,ID[k][0].astype(int))
    minSLP = np.append(minSLP,SLP[k][abs(hourstoSLPmin[k][0]).astype(int)])


pvival = np.array([-100,0.2,0.5,0.75,100])

anomaly = dict()
mature = dict()
layercounter = dict()
cycc =0

pinvals = np.arange(700,925.1,12.5)
plvlcounter = dict()

SPV = np.array([])
sLPV = np.array([])
PVmax = np.array([])
PVav = np.array([])
PVLav = np.array([])
PVmin = np.array([])
nT = np.array([])
nP  = np.array([])
SLPs = np.array([])
ids = np.array([])

split = ['cyc','env']
linestyle = ['-',':']
fsl=6
oroids = np.array([])
for ll,k in enumerate(datadi.keys()):
    q = np.where(avaID==int(k))[0][0]
    d = k
    
#    if (hourstoSLPmin[q][0]>-6):
#        continue

    PV = datadi[d]['PV'][:,0]
    P = datadi[d]['P'][:,0]
    PV[np.where(PV<-4)]=0
    i = np.where((PV>=0.75))[0]# & (P<925))[0]
    i2 = np.where(P<925)[0]
    ids = np.append(ids,k)
    SPV = np.append(SPV,np.sum(PV))
    sLPV = np.append(sLPV,np.sum(PV[i]))
    nP = np.append(nP,len(i2))
    nT = np.append(nT,len(i))
    PVmax = np.append(PVmax,np.max(PV))
    PVmin = np.append(PVmin,np.min(PV))
    PVav = np.append(PVav,np.mean(PV))
    PVLav = np.append(PVLav,np.mean(PV[i]))
    SLPs = np.append(SLPs,minSLP[q])

#    if np.mean(ORO[d]['env'][i,0])>0.2 and abs(np.mean(ORO[d]['cyc'][i,0]))<0.05: 
#     if np.mean(dipv[d]['env'][i,0])>0.3 and np.mean(dipv[d]['cyc'][i,0])>(-0.2):
#      if np.mean(ORO[d]['env'][i,0]/dipv[d]['env'][i,0])>0.65:
#       if np.mean(datadi[d]['OL'][i,0])<0.75:
#        if np.mean(ORO[d]['env'][i,12])/np.mean(ORO[d]['env'][i,0])>0.7:
    if k=='230058' or k=='241105':
        t = datadi[d]['time'][0]
        oroids = np.append(oroids,k)
        fig, axes = plt.subplots(2,1,sharex=True)
        axes = axes.flatten()
        qqq = len(i)
        for qq, ax in enumerate(axes[:2]):
            ax.plot([],[],marker=None,ls='-',color='black')
            ax.plot([],[],marker=None,ls=':',color='black')
            
            if qq==0:
                tmpdata = dipv[d]
            else:
                tmpdata = ORO[d]

            for pl, key in enumerate(split):
                ax.plot(t,np.mean(tmpdata[key][i],axis=0),color='k',ls=linestyle[pl])


            ax.axvline(hourstoSLPmin[q][0],color='grey',ls='-')
            ax.set_xlim(-48,0)
            ax.set_ylim(-0.3,1.5)
            ax.tick_params(labelright=False,right=True)
            ax.set_xticks(ticks=np.arange(-48,1,6))
        fig.text(0.5,0.94, dates[q][abs(hourstoSLPmin[q][0]).astype(int)] + '-' +d ,ha='center',fontsize=10)
        axes[0].set_ylabel('acc. PV [PVU]')
        axes[1].set_ylabel('topo. PV [PVU]')

        axes[1].legend(split,fontsize=fsl,loc='upper left')
#        i2 = np.where((PV>=0.75) & (ORO[k]['env'][:,0]/PV>=0.5))[0]
#        for iq in i2:
#            axes[2].plot(t,datadi[d]['P'][iq,:],color='grey',linewidth=0.1)

#        axes[2].set_ylabel('pressure [hPa]')
#        axes[2].set_ylim(1015,600)
#        axes[2].invert_yaxis()
        axes[1].set_xlabel('time until mature stage [h]')
        plt.subplots_adjust(left=0.1,bottom=None,top=None,right=None,hspace=0,wspace=0)
        name = 'oroPVsplit-' + dates[q][abs(hourstoSLPmin[q][0]).astype(int)] + '-ID-' + k + '.png'
        fig.savefig(psave + name,dpi=300,bbox_inches="tight")
        plt.close('all')

        fig, ax = plt.subplots()
        qqq = len(i)
        ax.plot([],[],marker=None,ls='-',color='black')
        ax.plot([],[],marker=None,ls=':',color='black')
        tmpdata = dipv[d]
        for pl, key in enumerate(split):
            ax.plot(t,np.mean(tmpdata[key][i],axis=0),color='k',ls=linestyle[pl])
            print(key,np.mean(tmpdata[key][i],axis=0)[0])


        ax.axvline(hourstoSLPmin[q][0],color='grey',ls='-')
        ax.set_xlim(-48,0)
        ax.set_ylim(-0.5,1.)
        ax.tick_params(labelright=False,right=True)
        ax.set_xticks(ticks=np.arange(-48,1,6))
        fig.text(0.5,0.94, dates[q][abs(hourstoSLPmin[q][0]).astype(int)] + '-' +d ,ha='center',fontsize=10)
        ax.set_ylabel('acc. PV [PVU]')

        ax.legend(split,fontsize=fsl,loc='upper left')
        ax.set_xlabel('time until mature stage [h]')
        name = 'PVsplit-' + dates[q][abs(hourstoSLPmin[q][0]).astype(int)] + '-ID-' + k + '.png'
        fig.savefig(psave + name,dpi=300,bbox_inches="tight")
        plt.close('all')


order = np.argsort(SLPs)
ids = ids[order]
SLPs2 = SLPs[order]
SPV = SPV[order]
sLPV = sLPV[order]
PVmax = PVmax[order]
PVav = PVav[order]
PVLav =PVLav[order]
PVmin =PVmin[order]
nT = nT[order]
nP  = nP[order]

#f = open(sp + 'mature-stages-pv-dict.txt',"rb")
#mature = pickle.load(f)
#f.close()
#f = open(sp + 'pv-anomaly-dict.txt',"rb")
#anomaly = pickle.load(f)
#f.close()
#f = open(sp + 'layer-count-dict.txt',"rb")
#closecounter = pickle.load(f)
#f.close()
#f = open(sp + 'gridpv-dict.txt',"rb")
#gridpv = pickle.load(f)
#f.close()
#
#f = open(sp + 'gridano-dict.txt',"rb")
#gridano = pickle.load(f)
#f.close()

#Pintervals = np.append(np.append(np.arange(100,550,25),np.arange(550,700,15)),np.arange(700,925.1,12.5))
#fulllayersanomaly =dict()
#for t in dates[maturestage]:
#    fulllayersanomaly[t] = np.zeros(len(Pintervals))
#
#
#clat = helper.radial_ids_around_center_calc(200)[1]
#totanomaly = np.zeros(len(dates[maturestage]))
#totPV = np.zeros(len(dates[maturestage]))
#realano = np.zeros(len(dates[maturestage]))
#gridsumpv = np.zeros(len(dates[maturestage]))
#for k,t in enumerate(dates[maturestage]):
#    for m in range(len(pinvals)-1):
#        realano[k] += np.sum(gridano[t][pinvals[m]])
#        gridsumpv[k] += np.sum(gridpv[t][pinvals[m]])
#
#    for l in range(len(clat)):
#        for m in range(len(pinvals)-1):
#            totPV[k] += mature[t][l,m]
#            totanomaly[k] += anomaly[t][l,m]
#
#
#order = np.argsort(relvort)
#realano = realano[order]
#gridsumpv = gridsumpv[order]
#
#relvort2 = relvort[order]
#totPV = totPV[order]
#totanomaly = totanomaly[order]
#
#fig, ax = plt.subplots()
##ax.plot(relvort2,totPV,color='k')
##ax.plot(relvort2,totanomaly,color='r')
#ax.plot(relvort2,gridsumpv,color='k',linestyle='-')
#ax.plot(relvort2,realano,color='r',linestyle='-')
#ax.set_xlabel(r'rel. vort. [$\times 10^{-4}$ s$^{-1}$]')
#ax.set_ylabel(r'$\Sigma$ PV [PVU]')
#
#fig.savefig(sp + 'low-level-anomaly-pv.png',dpi=300,bbox_inches="tight")
#plt.close('all')
#
#fig, ax = plt.subplots()
#ax.plot(relvort2,realano/gridsumpv,color='k')
#print(pearsonr(relvort2,realano/gridsumpv))
#ax.set_xlabel(r'rel. vort. [$\times 10^{-4}$ s$^{-1}$]')
#ax.set_ylabel(r' $\Sigma \Delta$ PV/$\Sigma$ PV')
#fig.savefig(sp+'low-level-anomaly-pv-percentage.png',dpi=300,bbox_inches="tight")
#plt.close('all')


#figg, gax = plt.subplots()
#lowano = dict()
#lowcounter = dict()
#
#fig, ax = plt.subplots()
#avanomaly = np.zeros(len(pinvals))
#for k,t in enumerate(dates[maturestage]): 
#    layeranomaly = np.zeros(len(pinvals))
#    lowano[t] = np.zeros(len(pinvals))
#    lowcounter[t] = np.zeros(len(pinvals))
#    for m in range(len(pinvals)-1):
# #       for l in range(len(clat)):
#        layeranomaly[m] += np.sum(gridano[t][pinvals[m]]) #anomaly[t][l,m]
#        lowano[t][m] += np.sum(gridano[t][pinvals[m]])
#        lowcounter[t][m] += len(gridano[t][pinvals[m]])
#        if closecounter[t][m]>0:
#            layeranomaly[m] /=closecounter[t][m]
#        avanomaly[m] += layeranomaly[m]
#    fulllayersanomaly[t][len(np.arange(100,550,25)) + len(np.arange(550,700,15)):] = layeranomaly
#    ax.plot(layeranomaly,pinvals,color=cm.bwr((relvort[k]-np.min(relvort))/(np.max(relvort)-np.min(relvort))))    
#
#avanomaly /= (k+1)
#ax.plot(avanomaly,pinvals,color='k')
#ax.set_ylim(700,925)
#ax.invert_yaxis()
#ax.set_xlabel(r'$\Delta$ PV [PVU]')
#ax.set_ylabel(r'pressure [hPa]')
#
#fig.savefig(sp + 'low-level-anomaly.png',dpi=300,bbox_inches="tight")
#plt.close()
#
#
#
#f = open(sp + 'mature-stages-pv-dict-upper.txt',"rb")
#mature = pickle.load(f)
#f.close()
#f = open(sp + 'pv-anomaly-dict-upper.txt',"rb")
#anomaly = pickle.load(f)
#f.close()
#f = open(sp + 'layer-count-dict-upper.txt',"rb")
#closecounter = pickle.load(f)
#f.close()
#f = open(sp + 'gridpv-dict-upper.txt',"rb")
#gridpv = pickle.load(f)
#f.close()
#f = open(sp + 'gridano-dict-upper.txt',"rb")
#gridano = pickle.load(f)
#f.close()
#
#pinvals = np.arange(100,550.1,25)
#clat = helper.radial_ids_around_center_calc(600)[1]
#
#upano = dict()
#upcounter = dict()
#
#fig, ax = plt.subplots()
#avanomaly = np.zeros(len(pinvals))
#for k,t in enumerate(dates[maturestage]):
#    layeranomaly = np.zeros(len(pinvals))
#    upano[t] = np.zeros(len(pinvals))
#    upcounter[t] = np.zeros(len(pinvals))
#    for m in range(len(pinvals)-1):
##        for l in range(len(clat)):
#        layeranomaly[m] += np.sum(gridano[t][pinvals[m]])#anomaly[t][l,m]
#        upano[t][m] += np.sum(gridano[t][pinvals[m]])
#        upcounter[t][m] += len(gridano[t][pinvals[m]])
#
#        if closecounter[t][m]>0:
#            layeranomaly[m] /=closecounter[t][m]
#        avanomaly[m] += layeranomaly[m]
#    fulllayersanomaly[t][:len(np.arange(100,550.1,25))] = layeranomaly
#    ax.plot(layeranomaly,pinvals,color=cm.bwr((relvort[k]-np.min(relvort))/(np.max(relvort)-np.min(relvort))))
#
#avanomaly /= (k+1)
#ax.plot(avanomaly,pinvals,color='k')
#ax.set_ylim(100,550)
#ax.invert_yaxis()
#ax.set_xlabel(r'$\Delta$ PV [PVU]')
#ax.set_ylabel(r'pressure [hPa]')
#
#fig.savefig(sp + 'upper-level-anomaly.png',dpi=300,bbox_inches="tight")
#plt.close()
#
#
#f = open(sp + 'mature-stages-pv-dict-mid.txt',"rb")
#mature = pickle.load(f)
#f.close()
#f = open(sp + 'pv-anomaly-dict-mid.txt',"rb")
#anomaly = pickle.load(f)
#f.close()
#f = open(sp + 'layer-count-dict-mid.txt',"rb")
#closecounter = pickle.load(f)
#f.close()
#f = open(sp + 'gridpv-dict-mid.txt',"rb")
#gridpv = pickle.load(f)
#f.close()
#f = open(sp + 'gridano-dict-mid.txt',"rb")
#gridano = pickle.load(f)
#f.close()
#
#pinvals = np.arange(550,700.1,15)
#
#clat = helper.radial_ids_around_center_calc(200)[1]
#midano = dict()
#midcounter = dict()
#midlayerperc = dict()
#
#fig, ax = plt.subplots()
#
#avanomaly = np.zeros(len(pinvals))
#
#for k,t in enumerate(dates[maturestage]):
#    layeranomaly = np.zeros(len(pinvals))
#    midano[t] = np.zeros(len(pinvals))
#    midcounter[t] = np.zeros(len(pinvals))
#    midlayerperc[t] = np.zeros(len(pinvals))
#    for m in range(len(pinvals)-1):
##        for l in range(len(clat)):
##            layeranomaly[m] += anomaly[t][l,m]
#        layeranomaly[m] += np.sum(gridano[t][pinvals[m]])
#        midano[t][m] += np.sum(gridano[t][pinvals[m]])
#        midcounter[t][m] += len(gridano[t][pinvals[m]])
#        if closecounter[t][m]>0:
#            layeranomaly[m] /=closecounter[t][m]
#        avanomaly[m] += layeranomaly[m]
#
#    fulllayersanomaly[t][len(np.arange(100,550,25)):-len(np.arange(700,925.1,12.5))] = layeranomaly[:-1]
#    ax.plot(layeranomaly,pinvals,color=cm.bwr((relvort[k]-np.min(relvort))/(np.max(relvort)-np.min(relvort))))
#
#avanomaly /= (k+1)
#ax.plot(avanomaly,pinvals,color='k')
#ax.set_ylim(550,700)
#ax.invert_yaxis()
#ax.set_xlabel(r'$\Delta$ PV [PVU]')
#ax.set_ylabel(r'pressure [hPa]')
#
#fig.savefig(sp + 'mid-level-anomaly.png',dpi=300,box_inches="tight")
#plt.close()
#
#
#totano = np.zeros(len(maturestage))
#for k,t in enumerate(dates[maturestage]):
#    gax.plot(fulllayersanomaly[t],Pintervals,color=cm.bwr((relvort[k]-np.min(relvort))/(np.max(relvort)-np.min(relvort))))
#gax.set_ylim(200,925)
#gax.set_xlim(-0.5,3)
#gax.invert_yaxis()
#gax.set_xlabel(r'$\Delta$ PV [PVU]')
#gax.set_ylabel(r'pressure [hPa]')
#
#figg.savefig(sp + 'all-level-anomaly.png',dpi=300,bbox_inches="tight")
#plt.close()    
#summedanos = dict()
#counters = dict()
#summedDeltaPV = np.zeros(len(maturestage))
#
#layerpercentage = dict()
#
#fig, ax = plt.subplots()
#
#for k,t in enumerate(dates[maturestage]):
#    summedanos[t] = np.append(np.append(upano[t][4:],midano[t][:-1]),lowano[t][:-1])
#    counters[t] = np.append(np.append(upcounter[t][4:],midcounter[t][:-1]),lowcounter[t][:-1])
#    summedDeltaPV[k] = np.sum(summedanos[t])
#    layerpercentage[t] = np.flip(np.cumsum(np.flip(summedanos[t])))/summedDeltaPV[k]
#    ax.plot(layerpercentage[t],Pintervals[4:],color=cm.bwr((relvort[k]-np.min(relvort))/(np.max(relvort)-np.min(relvort))))
##    print(relvort[k],np.sum(upano[t])/summedDeltaPV[k],np.sum(midano[t][:-1])/summedDeltaPV[k],np.sum(lowano[t][:-1])/summedDeltaPV[k])
##    print(np.sum(upano[t]),np.sum(midano[t]),np.sum(lowano[t]),summedDeltaPV[k])
#
#ax.set_ylim(200,925)
#ax.invert_yaxis()
#ax.set_xlabel(r'layer contribution to $\Delta$ PV [%]')
#ax.set_ylabel(r'pressure [hPa]')
#fig.savefig(sp + 'layer-ano-percentage.png',dpi=300,bbox_inches="tight")
#plt.close('all')
#


