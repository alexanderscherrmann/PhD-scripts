import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

X = '300-'
X=''
which = ['weak-' + X + 'cyclones.csv','moderate-' + X + 'cyclones.csv','intense-' + X + 'cyclones.csv']
when = ['onedaypriormature','twodaypriormature','threedaypriormature','fourdaypriormature','fivedaypriormature','sixdaypriormature','sevendaypriormature']
labels = np.arange(1,8)

dp = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
if X=='':
    X='100-'
ylim = int(X[:-1])/5 + 10

x = np.arange(0,8)
regimes = ['no','AT','GL','AR','ZOEA','ZOWE','BL','ZO']



cmap = matplotlib.cm.terrain
slplevels = np.arange(970,1030.1,0.5)
norm = plt.Normalize(970,1030)

for l,we in zip(labels,when):

    fig, axes = plt.subplots(1,3,sharey=True)
    plt.subplots_adjust(wspace=0.1,hspace=0,top=0.5)
    axes=axes.flatten()
    for ax,wi in  zip(axes,which):
        slps = np.array([])
        sel = pd.read_csv(dp + wi)
        DJF = np.where((sel['month'].values==1) | (sel['month'].values==2) | (sel['month'].values==12))[0]
        for wrm in range(0,8):
            slps = np.append(slps, np.mean(sel['minSLP'].values[DJF][np.where(sel[we+'-WR'].values[DJF]==wrm)[0]]))

        ax.hist(sel[we + '-WR'].values[DJF],bins=8,range=[0,8],facecolor='grey',edgecolor='k',alpha=1)
        ax2=ax.twinx()
        ax2.scatter(np.arange(0.5,8),slps,color='b',marker='*')
        ax2.set_ylim(970,1030)
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
    fig.savefig(pi + 'WR-hist-'+ X + we +'-DJF.png',dpi=300,bbox_inches='tight')
    plt.close('all')


