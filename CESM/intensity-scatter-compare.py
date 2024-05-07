import sys
sys.path.append('/home/ascherrmann/scripts/')
import wrfsims
import numpy as np
import matplotlib.pyplot as plt
import pickle

sim,at,med=wrfsims.cesm_ids()

tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'
sea='DJF'

colors=['saddlebrown','k','purple','red','b']
names = ['-0-km','west','east','south','north']


xbase = [0,1,1.6,2.2]
xchange = [-0.12,0,0.12]
km=['-0-km','200-km','400-km','800-km']
period=['ERA5','2010','2040','2070','2100']
amps=['0.7','1.4','2.1']
markers=['*','s','o']
labels=['(a) ERA5','(b) CESM-2010','(c) CESM-2040','(d) CESM-2070','(e) CESM-2100']

fig,axes = plt.subplots(1,5,figsize=(16,5),sharey=True)
ax = axes[0]
for co in colors:
    ax.plot([],[],marker='o',ls='',color=co)

for ma in markers:
    ax.plot([],[],marker=ma,ls='',color='grey')
for q,perio in enumerate(period):

    ax = axes[q]
    for x,k in zip(xbase[1:],km[1:]):
        for mark,xc,amp in zip(markers,xchange,amps):
            xp=x+xc
            for co,na in zip(colors,names):
                if na=='-0-km':

                    sim='CESM-%s-%s%s-max-%s-QGPV'%(perio,sea,km[0],amp)
                else:
                    sim='CESM-%s-%s-%s-%s-%s-QGPV'%(perio,sea,k,na,amp)

                try:
                    tra = np.loadtxt(tracks+sim+'-new-tracks.txt')
                except:
                    continue
                slp = tra[:,3]
                IDs = tra[:,-1]
                loc = np.where(IDs==2)[0]
                ax.scatter(xp,np.min(slp[loc]),color=co,marker=mark)

    ax.set_ylim(990,1020)
    ax.set_xticks(xbase[1:])
    ax.set_xticklabels(['200km','400km','800km'])
    ax.text(0.01,1.02,labels[q],transform=ax.transAxes)


axes[0].legend(['Center','West','East','South','North','0.7','1.4','2.1'],loc='upper left',ncol=2)
axes[0].set_ylabel('minimum SLP [hPa]')
plt.subplots_adjust(wspace=0,hspace=0)
#    ax.plot([],[],marker='o',ls='',color=co)

#labels=np.array(['ERA5\n1979-2020','ERA5\n1981-2010','CESM\n1981-2010','CESM\n2011-2040','CESM\n2041-2070','CESM\n2071-2100'])

fig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/images/SLP-scatter-compare.png',dpi=300,bbox_inches='tight')
plt.close('all')
