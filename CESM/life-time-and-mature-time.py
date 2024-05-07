import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import wrfsims
import pickle

#SIMS,ATIDS,MEDIDS = wrfsims.cesm_ids_wrongSKIN()
SIMS,ATIDS,MEDIDS = wrfsims.cesm_ids()
SIMS = np.array(SIMS)

tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

pappath = '/atmosdyn2/ascherrmann/015-CESM-WRF/'
dwrf='/atmosdyn2/ascherrmann/013-WRF-sim/'
se='DJF'
period=['ERA5','2010','2040','2070','2100']

gentime=dict()
lysistime=dict()
maturetime=dict()
lifetime=dict()
amps = ['0.7','1.4','2.1']
for perio in period:
 gentime[perio] = dict()
 lysistime[perio]=dict()
 maturetime[perio]=dict()
 lifetime[perio]=dict()
 for amp in ['0.7','1.4','2.1']:
 
  gt=np.array([])
  lt=np.array([])
  lft=np.array([])
  mt=np.array([])

  agt=np.array([])
  alt=np.array([])
  alft=np.array([])
  amt=np.array([])

  for simid,sim in enumerate(SIMS):
     if not amp in sim or 'wrong' in sim or not perio in sim:
         continue
     
     medid = np.array(MEDIDS[simid])
 
     try:
         tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
     except:
         continue

     t = tra[:,0]
     tlon,tlat = tra[:,1],tra[:,2]
     slp = tra[:,3]
     IDs = tra[:,-1]
     loc = np.where(IDs==2)[0]

     gt=np.append(gt,t[loc[0]]/24)
     lt=np.append(lt,t[loc[-1]]/24)
     lft=np.append(lft,lt[-1]-gt[-1])
     mt=np.append(mt,t[loc[np.where(slp[loc]==np.min(slp[loc]))[0][0]]]/24)

     loc = np.where(IDs==1)[0]

     agt=np.append(agt,t[loc[0]]/24)
     alt=np.append(alt,t[loc[-1]]/24)
     alft=np.append(alft,lt[-1]-gt[-1])
     amt=np.append(amt,t[loc[np.where(slp[loc]==np.min(slp[loc]))[0][0]]]/24)

  gentime[perio][amp] = gt
  lysistime[perio][amp] = lt
  lifetime[perio][amp] = lft
  maturetime[perio][amp] = mt  
     
SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)
perio='ERA5 42y'

gentime[perio] = dict()
lysistime[perio]=dict()
maturetime[perio]=dict()
lifetime[perio]=dict()
for amp in ['0.7','1.4','2.1']:
  gt=np.array([])
  lt=np.array([])
  lft=np.array([])
  mt=np.array([])

  agt=np.array([])
  alt=np.array([])
  alft=np.array([])
  amt=np.array([])

  for simid,sim in enumerate(SIMS):
     if not amp in sim or 'wrong' in sim or 'not' in sim or 'check' in sim or 'AT' in sim or not 'DJF' in sim:
         continue
     medid = np.array(MEDIDS[simid])

     try:
         tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
     except:
         continue

     t = tra[:,0]
     tlon,tlat = tra[:,1],tra[:,2]
     slp = tra[:,3]
     IDs = tra[:,-1]
     loc = np.where(IDs==2)[0]

     gt=np.append(gt,t[loc[0]]/24)
     lt=np.append(lt,t[loc[-1]]/24)
     lft=np.append(lft,lt[-1]-gt[-1])
     mt=np.append(mt,t[loc[np.where(slp[loc]==np.min(slp[loc]))[0][0]]]/24)

     loc = np.where(IDs==1)[0]

     agt=np.append(agt,t[loc[0]]/24)
     alt=np.append(alt,t[loc[-1]]/24)
     alft=np.append(alft,lt[-1]-gt[-1])
     amt=np.append(amt,t[loc[np.where(slp[loc]==np.min(slp[loc]))[0][0]]]/24)

  gentime[perio][amp] = gt
  lysistime[perio][amp] = lt
  lifetime[perio][amp] = lft
  maturetime[perio][amp] = mt


save = dict()
save['gensistime'] = gentime
save['lysistime'] = lysistime
save['lifetime'] = lifetime
save['maturetime'] = maturetime

f = open(dwrf + 'data/life-timedata-%s.txt'%se,'wb')
pickle.dump(save,f)
f.close()


fig,axes=plt.subplots(2,2,sharex=True)

fig,axes=plt.subplots(figsize=(9,6),nrows=2,ncols=2,sharex=True)
axes = axes.flatten()
flier = dict(marker='+',markerfacecolor='grey',markersize=1,linestyle=' ',markeredgecolor='grey')
meanline = dict(linestyle='-',linewidth=1,color='red')
meanline2 = dict(linestyle='-',linewidth=1,color='dodgerblue')
capprops = dict(linestyle='-',linewidth=1,color='grey')
medianprops = dict(linestyle='-',linewidth=1,color='black')
boxprops = dict(linestyle='-',linewidth=1.,color='slategrey')
whiskerprops= dict(linestyle='-',linewidth=1,color='grey')

labels=np.array(['E42','E2010','C2010','C2040','C2070','C2100'])

periods = ['ERA5 42y', 'ERA5','2010','2040','2070','2100' ]
ymin,ymax=[2,5.5,2,3.5],[6.5,9.5,6.5,9.5]
for pl,ylab,(q,wh) in zip(['(a)','(b)','(c)','(d)'],['gensis time [d]','lysis time [d]','life time [d]', 'mature stime [d]'],enumerate(['gensistime','lysistime','lifetime','maturetime'])):
 meanbox=[]

 for perio in periods:
    tmp = np.array([])
    for amp in amps:
        tmp = np.append(tmp,save[wh][perio][amp])
    meanbox.append(tmp)

 ax = axes[q]
 bp = ax.boxplot(meanbox,whis=(10,90),labels=labels,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops)
 ax.set_ylabel(ylab)
 ax.set_xlim(0.5,6.5)
 ax.set_ylim(ymin[q],ymax[q])
 ax.text(0.01,0.92,pl,transform=ax.transAxes)
fig.subplots_adjust(hspace=0)
fig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/images/box-plots-lifetimes.png',dpi=300,bbox_inches='tight')
plt.close('all')




