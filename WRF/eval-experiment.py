import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as pr

f = open('/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/test-PV-distance.txt','rb')
d = pickle.load(f)
f.close()
ght = d['ght']
r = d['r']
PVdi = d['pv']
w = ['f','h','m']
ghtrefo = d['ghtref850']
rr = d['rref']
p = d['p']


PV = dict()
stat = dict()
presdi = dict()
ghtdi = dict()
ghtref = dict()
ghtghtref = dict()
for l in ['low','upp']:
  fig,axes = plt.subplots(1,3,sharey=True)
  stat[l] = dict()
  presdi[l] = dict()
  PV[l] = dict()
  ghtdi[l] = dict()
  ghtref[l] = dict()
  ghtghtref[l]=  dict()

  for k in w:
      stat[l][k] = np.zeros(len(list(PVdi[l].keys())))
      ghtdi[l][k] = np.zeros(len(list(PVdi[l].keys())))
      presdi[l][k] = np.zeros(len(list(PVdi[l].keys())))
      ghtref[l][k] = np.zeros(len(list(PVdi[l].keys())))
      ghtghtref[l][k] = np.zeros(len(list(PVdi[l].keys())))

  for wq,i in enumerate(PVdi[l].keys()):
    PV[l][i]=dict()
    
    for qk,k,ax in zip(range(3),w,axes):

        PV[l][i][k] = PVdi[l][i][k] * r[l][i][k]
        
        loc = np.where(PV[l][i][k]>=0.75)[0]
        avmeasure = np.mean(PVdi[l][i][k][loc]*r[l][i][k][loc]/rr[l][i][k][loc] * np.mean(rr[l][i][k][loc]))
#        avmeasure = np.mean(newpvrefdi[l][i][k][loc] * np.mean(newrrefdi[l][i][k][loc]))
        stat[l][k][wq] = avmeasure
        pres = np.mean(p[l][i][k][loc]*PV[l][i][k][loc]/np.mean(PV[l][i][k][loc]))
        #ghttr = np.mean(PV[l][i][k][loc]/ghtrefo[l][i][k][loc] * np.mean(ghtrefo[l][i][k][loc]))
        #ghttr = np.mean(PV[l][i][k][loc]/ght[l][i][k][loc] * np.mean(ght[l][i][k][loc]))
        ghttr = np.mean(ght[l][i][k][loc] * PV[l][i][k][loc]/np.mean(PV[l][i][k][loc]))
        ax.scatter(d['slp'][wq,qk],avmeasure,color='k')
        presdi[l][k][wq] = pres
        ghtdi[l][k][wq] =  ghttr
        ghtref[l][k][wq] = np.mean(np.unique(ghtrefo[l][i][k][loc]))      
        ghtghtref[l][k][wq] = np.mean(ght[l][i][k][loc]-ghtrefo[l][i][k][loc])

  if l=='low':
      axes[0].set_ylim(0,10)

  axes[0].set_ylabel('some PV [PVU]')
  axes[0].set_xlabel('genesis SLP [hPa]')
  axes[1].set_xlabel('half mature SLP [hPa]')
  axes[2].set_xlabel('mature SLP [hPa]')
  fig.savefig('/atmosdyn2/ascherrmann/013-WRF-sim/image-output/experimental-%s.png'%l, dpi=300,bbox_inches='tight')
  plt.close('all')

for l in ['upp','low']:
 for q,k in enumerate(w):
    print(k,pr(d['slp'][:,q],stat[l][k]))

fig,axes =plt.subplots(1,3,sharey=True)
for qk,k,ax in zip(range(3),w,axes):
    ax.scatter(stat['low'][k],stat['upp'][k],color='k')
    ax.set_xlim(0,10)
    print(pr(stat['low'][k],stat['upp'][k]))
fig.savefig('/atmosdyn2/ascherrmann/013-WRF-sim/image-output/experimental-ul.png', dpi=300,bbox_inches='tight')
plt.close('all')

for nam,var,ylab in zip(['pressure','GHT','GHT-ref','GHT-GHTref'],[presdi,ghtdi,ghtref,ghtghtref],['PV weighted pressure [hPa]','PV weighted GHT [m]','GHT @850 hPa [m]','GHT-GHT@850hPa [m]']):
 for l in ['upp','low']:
    fig,axes =plt.subplots(1,3,sharey=True)
    for q,k in enumerate(w):
        print(nam,k,pr(d['slp'][:,q],var[l][k]))
        axes[q].scatter(d['slp'][:,q],var[l][k],color='k')
        #axes[q].set_ylim(0,np.percentile(var[l][k],90) * 1.25)
    axes[0].set_ylabel(ylab)
    fig.savefig('/atmosdyn2/ascherrmann/013-WRF-sim/image-output/experimental-%s-%s.png'%(l,nam),dpi=300,bbox_inches='tight')
    plt.close('all')








