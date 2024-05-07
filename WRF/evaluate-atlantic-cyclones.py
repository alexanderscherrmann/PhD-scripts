import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle

import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec


ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'


f = open(ps + 'Atlantic-cyclones-prior-intense-moderate-weak-MED-cyclones.txt','rb')
data = pickle.load(f)
f.close()

whenplot = ['threedaypriormature','fourdaypriormature','fivedaypriormature','sixdaypriormature','sevendaypriormature']#,'threedaypriortrack0','fourdaypriortrack0','fivedaypriortrack0','sixdaypriortrack0','sevendaypriortrack0']
savelabel = ['m3','m4','m5','m6','m7']#,'t3','t4','t5','t6','t7']

minlon = -90
minlat = 10
maxlat = 80
maxlon = -20

histvar = ['currentSLP','age','currenthtSLPmin','lifetime','minSLP']
ranges = [[965,1010],[0,144],[-60,60],[48,240],[965,1010]]
bins = [32,36,30,48,32]
xlab = ['current SLP [hPa]','age [h]','time to minSLP [h]','lifetime [h]','min SLP [hPa]']
xticks = [np.arange(965,1011,5),np.arange(0,145,24),np.arange(-60,61,12),np.arange(48,241,24),np.arange(965,1011,5)]
colm = ['weak','moderate','intense']

for xla,his,r,b,xtick in zip(xlab,histvar,ranges,bins,xticks):
 for we in data['DJF']['weak-cyclones.csv'][50].keys():
    if np.any(np.array(whenplot)==we):
        sl = savelabel[np.where(np.array(whenplot)==we)[0][0]]
        for ll in [50]:#, 100, 150, 200]:
            fig,axes = plt.subplots(4,3,sharex=True,sharey=True)
            for q,axe,sea in zip(range(len(axes)),axes,['DJF','MAM','JJA','SON']):
                axe[0].text(0.05,0.9,sea,transform=axe[0].transAxes,fontsize=6,fontweight='bold')

                for c,ax, wi in zip(colm,axe,['weak-cyclones.csv','moderate-cyclones.csv','intense-cyclones.csv']):
                    ax.hist(data[sea][wi][ll][we][his],bins=b,range=r,facecolor='grey',edgecolor='k',alpha=1)
                    
                    ax.set_xlim(r)
                    ax.set_ylim(0,0.2*ll)
                    ax.set_xticks(ticks=xtick)
                    if q==0:
                        ax.text(0.4,1.1,c,transform=ax.transAxes,fontsize=6,fontweight='bold')

                    if sea=='SON':
                        ax.set_xlabel(xla)
                        for label in ax.xaxis.get_ticklabels()[1::2]:
                            label.set_visible(False)
                        

            plt.subplots_adjust(wspace=0.05,hspace=0)
            name = pi + 'pre-atlantic-cyclones/' + his + '-%03d-'%ll + f'{sl}.png' 
            fig.savefig(name,dpi=300,bbox_inches='tight')
            plt.close(fig)



color =['blue','palegreen','red','saddlebrown']


for wi in data['DJF'].keys():
    for ll in [50]:#, 100, 150, 200]:
        for we in data['DJF']['weak-cyclones.csv'][50].keys():
            if np.any(np.array(whenplot)==we):
                sl = savelabel[np.where(np.array(whenplot)==we)[0][0]]
                fig = plt.figure(figsize=(6,4))
                gs = gridspec.GridSpec(ncols=1, nrows=1)
                ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
                ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
                ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
                ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())
        
                hand = []
                for sea,co in zip(data.keys(),color):
                    hand.append(ax.scatter(data[sea][wi][ll][we]['lon'],data[sea][wi][ll][we]['lat'],marker='.',color=co))

                ax.legend(hand,list(data.keys()),loc='upper right')
                name = pi + 'pre-atlantic-cyclones/' + 'current-location-%03d-'%ll + f'{sl}.png'
                fig.savefig(name,dpi=300,bbox_inches='tight')
                plt.close(fig)


