import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper

from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm


f = open('/home/ascherrmann/010-IFS/ctraj/MED/use/PV-data-MEDdPSP-100-ZB-800PVedge-0.3-400-correct-distance.txt','rb')
data =pickle.load(f)
f.close()

datadi = data['rawdata']
dipv = data['dipv']

CID = '20171214_02-073'

lon0 = datadi[CID]['lon'][:,-1]
lat0 = datadi[CID]['lat'][:,-1]

reg = [[5,46,15,90],
       [5,25,13,37],
       [5,38,21,46.5],
       [14,29,28,41]]

hoursegments = np.flip(np.arange(-48,1,1))
linewidth=1.5
alpha=1.
cmap = ListedColormap(['saddlebrown','orange'])
norm = BoundaryNorm([0, 0.5, 1], cmap.N)
fsl=6

cl=['k','orange','saddlebrown','green','dodgerblue','blue','red']
pllegend = ['cyc','env','TOT','|CONVT|>|TURBT|','|TURBT|>|CONVT|', 'CONVM', 'TURBM','RAD','LS']

fig, axes = plt.subplots(2,2,sharex=True,sharey=True)
plt.subplots_adjust(left=0.1,wspace=0,hspace=0)

axes = axes.flatten()
for r,ax in zip (reg,axes):
    ax.plot([],[],color='k',ls='-')
    ax.plot([],[],color='k',ls=':')
    for c in cl:
        ax.plot([],[],color=c,ls='-')

    traid = np.where((lon0>=r[0]) & (lon0<=r[2]) & (lat0>=r[1]) & (lat0<=r[-1]))[0]

    for pro,col in zip(['PVR-T','APVTOT','PVRLS','APVRAD','PVRCONVM','PVRTURBM'],['orange','k','r','b','g','dodgerblue']):

        for ls,key in zip(['-',':'],['cyc','env']):
            meantmp = np.mean(dipv[CID][key][pro][traid],axis=0)
            if pro=='PVR-T':
                segmentval = np.array([])
                for xx in range(49):
                    if (abs(np.sum(datadi[CID]['PVRCONVT'][traid,xx]))>=(abs(np.sum(datadi[CID]['PVRTURBT'][traid,xx])))):
                        segmentval = np.append(segmentval,1)
                    else:
                        segmentval = np.append(segmentval,0)

                segments = helper.make_segments(hoursegments,meantmp)
                lc = mcoll.LineCollection(segments, array=segmentval, cmap=cmap, norm=norm, linestyle=ls,linewidth=linewidth, alpha=alpha)
                ax.add_collection(lc)
            else:
                ax.plot(datadi[CID]['time'][0],meantmp,color=col,linestyle=ls)


    ax.set_xlim(-48,0)
    ax.set_ylim(-0.3,1.8)
    ax.tick_params(labelright=False,right=True)

axes[2].set_xticks(ticks=np.arange(-48,1,6))
axes[3].set_xticks(ticks=np.arange(-42,1,6))

axes[0].legend(pllegend,fontsize=fsl,loc='upper left',frameon=False)

axes[0].set_ylabel('acc. PV [PVU]')
axes[2].set_ylabel('acc. PV [PVU]')
axes[2].set_xlabel('time until mature stage [h]')
axes[3].set_xlabel('time until mature stage [h]')

fig.savefig('/home/ascherrmann/010-IFS/trajectory-origin-apv.png',dpi=300,bbox_inches='tight')
plt.close('all')
