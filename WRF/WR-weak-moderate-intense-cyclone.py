import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = '300-'
#X=''
which = ['weak-' + X + 'cyclones.csv','moderate-' + X + 'cyclones.csv','intense-' + X + 'cyclones.csv']
when = ['onedaypriormature','twodaypriormature','threedaypriormature','fourdaypriormature','fivedaypriormature','sixdaypriormature','sevendaypriormature']
labels = np.arange(1,8)

dp = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
if X=='':
    X='100-'
ylim = int(X[:-1])/5 + 10
#freq = dict()
#for wi in which:
#    freq[wi] = dict()
#    sel = pd.read_csv(dp + wi)
#    for we in when:
#        tmp = np.array([])
#        R = sel[we + '-WR']
#        for k in range(0,8):
#            tmp = np.append(tmp,len(np.where(R==k)[0]))
#        freq[wi][we] = tmp

x = np.arange(0,8)
regimes = ['no','AT','GL','AR','ZOEA','ZOWE','BL','ZO']

for l,we in zip(labels,when):

    fig, axes = plt.subplots(1,3,sharey=True)
    plt.subplots_adjust(wspace=0.1,hspace=0,top=0.5)
    axes=axes.flatten()
    
    for ax,wi in  zip(axes,which):
        sel = pd.read_csv(dp + wi)
        ax.hist(sel[we + '-WR'],bins=8,range=[0,8],facecolor='grey',edgecolor='k',alpha=1)
        ax.set_xticks(ticks=x+0.5)
        ax.set_xticklabels(labels=regimes)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_ylim(0,ylim)
        if X=='100-':
            ax.text(0.4,0.95,wi[:-(12+1)],transform=ax.transAxes,fontsize=8,fontweight='bold')
        else:
            ax.text(0.4,0.95,wi[:-(len(X)+12+1)],transform=ax.transAxes,fontsize=8,fontweight='bold')
    axes[0].set_ylabel('counts')
    axes[0].text(0,0.95,'-%d d'%l,transform=axes[0].transAxes,fontsize=8,fontweight='bold')
    fig.savefig(pi + 'WR-hist-'+ X +  we +'.png',dpi=300,bbox_inches='tight')
    plt.close('all')
