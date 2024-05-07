import numpy as np
import pickle
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import helper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patch
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import cm
import argparse
import cartopy
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser(description="plot accumulated average PV gain that is associated with the cyclone and the environment")
parser.add_argument('rdis',default='',type=int,help='distance from center in km that should be considered as cyclonic')
args = parser.parse_args()
rdis = int(args.rdis)

pload = '/home/ascherrmann/009-ERA-5/MED/ctraj/use/'
pload2 = '/home/ascherrmann/009-ERA-5/MED/traj/use/'
f = open(pload + 'PV-data-dPSP-100-ZB-800-2-%d-correct-distance.txt'%rdis,'rb')
PVdata = pickle.load(f)
f.close()

dipv = PVdata['dipv']
datadi = PVdata['rawdata']
dit = PVdata['dit']

both = np.array([])
adv = np.array([])
cyclonic = np.array([])
environmental = np.array([])

savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']
var = []

for u,x in enumerate(savings):
    f = open(pload2[:-4] + x + 'furthersel.txt',"rb")
    var.append(pickle.load(f))
    f.close()

SLP = var[0]
lon = var[1]
lat = var[2]
ID = var[3]
hourstoSLPmin = var[4]
dates = var[5]
avaID = np.array([])
maturedates = np.array([])
for k in range(len(ID)):
    avaID=np.append(avaID,ID[k][0].astype(int))
    maturedates = np.append(maturedates,dates[k][abs(hourstoSLPmin[k][0]).astype(int)])

adv = np.array([])
cyc = np.array([])
env = np.array([])
c = 'cyc'
e = 'env'

PVstart = np.array([])
PVend = np.array([])
ca = 0
ct = 0
pvloc = dict()
cycd = dict()
envd = dict()

hcyc=12
f = open('/home/ascherrmann/009-ERA-5/MED/counter.txt','rb')
count = pickle.load(f)
f.close()
counters=dict()
for k in [60,85]:
    counters[k] = count[k]

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(nrows=2,ncols=3)
axes = []
for k in range(2):
    for l in range(3):
        axes.append(fig.add_subplot(gs[k,l],projection=ccrs.PlateCarree()))
        
LON = np.arange(-180,180.1,0.5)
LAT = np.arange(-90,90.1,0.5)

LON2 = np.arange(-180,180.1,1)
LAT2 = np.arange(-90,90.1,1)
ncounter = dict()
nc = dict()
nc2 = dict()
for k in [60,85]:
    ncounter[k] = dict()
    nc[k] = dict()
    nc2[k] = dict()
    for q in counters[k].keys():
        print(k,q,np.sum(counters[k][q]))
        ncounter[k][q] = np.zeros((len(LAT2),len(LON2)))
        nc[k][q] = np.zeros((len(LAT),len(LON)))
        nc2[k][q] = np.zeros((len(LAT2),len(LON2)))
        for u in range(len(LON)):
            for e in range(len(LAT)):
                nc[k][q][e,u] = np.mean(counters[k][q][e-2:e+3,u-2:u+3])
        for u in range(len(LON2)):
            for e in range(len(LAT2)):
                ncounter[k][q][e,u] = np.sum(counters[k][q][2*e:2*e+2,2*u:2*u+2])
        for u in range(len(LON2)):
            for e in range(len(LAT2)):
                nc2[k][q][e,u] = np.mean(ncounter[k][q][e-1:e+2,u-1:u+2])



#count2 = dict()
#for k in [60,85]:
#    count2[k] = dict()
#    for q in range(0,6):
#        count2[k][q]=np.zeros((361,721))
#        for l in range(0,361):
#            for a in range(0,721):
#                count2[k][q][l,a] = np.mean(counters[k][q][l-1:l+2,a-1:a+2])
#

minpltlonc = -10
maxpltlonc = 45
minpltlatc = 25
maxpltlatc = 50
steps = 5

lonticks=np.arange(minpltlonc, maxpltlonc+1,steps*3)
latticks=np.arange(minpltlatc, maxpltlatc+1,steps*2)

#lab = ['60% cyc', '60% env','60% adv', '75% cyc','75%env', '75% adv','85% cyc','85% env','85% adv','90% cyc','90% env','90% adv']
labs = ['cyc','env','adv']
labels = ['a)','b)','c)','d)','e)','f)','g)','h)','i)','j)','k)','l)']
colo = ['blue','red']

levels = np.arange(1,16,2)
colors=['b','red']
for q, ax in enumerate(axes):
#    if q==0:
#        ax.plot([],[],ls=' ',marker='.',color='k',markersize=2)
    #if q==1:
#        ax.plot([],[],ls=' ',marker='.',color='k',markersize=28/3.5)
#    print(np.max(counters[60][q]),np.max(counters[85][q]))
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)
    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc])
    for u,k in enumerate([60,85]):
        latids,lonids = np.where(counters[k][q]!=0)
        
#        latids,lonids = np.where(counters[k][q]!=0)
#        ax.scatter(LON2[lonids],LAT2[latids],color=colo[u],s=ncounter[k][q][latids,lonids]/2)
#       ax.scatter(LON[lonids],LAT[latids],color=colo[u],s=counters[k][q][latids,lonids]/2)
#       ax.contour(LON,LAT,counters[k][q],linewidths=0.5,colors=colors[u],levels=levels)
        if q<=3:
            ax.contour(LON2,LAT2,nc2[k][q],linewidths=0.5,colors=colors[u],levels=levels)
        else:
            ax.scatter(LON[lonids],LAT[latids],color=colo[u],s=counters[k][q][latids,lonids]/2)
#       cf = ax.contour(LON,LAT,nc[k][q],linewidths=0.5,colors=colors[u],levels=levels) 
#        if q<4:
#            if q==3 and k==85:
#                latids,lonids = np.where(counters[k][q]!=0)
#                ax.scatter(LON[lonids],LAT[latids],color=colo[u],s=counters[k][q][latids,lonids]/2)
#            else:
#                cf = ax.contour(LON,LAT,count2[k][q],linewidths=0.5,colors=colors[u],levels=levels)
#        else:
#            latids,lonids = np.where(counters[k][q]!=0)
#            ax.scatter(LON[lonids],LAT[latids],color=colo[u],s=counters[k][q][latids,lonids]/2)

#    ax.coastlines()
    ax.set_aspect('auto')
    ax.set_xticks(lonticks, crs=ccrs.PlateCarree())
    ax.set_yticks(latticks, crs=ccrs.PlateCarree())
#    if q%3==0:
    ax.set_yticklabels(labels=latticks,fontsize=8)
    ax.yaxis.set_major_formatter(LatitudeFormatter())
#    if q>=3:
    ax.set_xticklabels(labels=lonticks,fontsize=8)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    if q%3!=0:
        ax.set_yticklabels([])

    ax.text(0.01, 0.98, labels[q], transform=ax.transAxes,fontsize=12, fontweight='bold',va='top')


#axes[0].legend(['1 cyclone','30 cyclones'],loc='lower right',frameon=True,fontsize='x-small',ncol=2)
#axes[1].legend([' 30 cyclones'],loc='lower right',frameon=False,fontsize='small')

for ax in [axes[1],axes[2],axes[4],axes[5]]:
    ax.set_yticklabels([])

plt.subplots_adjust(bottom=0.1,top=0.6,wspace=0,hspace=0.15)
#
##cax = fig.add_axes([0, 0, 0.1, 0.1])
##cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,orientation='vertical',extend='both')
##cbar.ax.set_yticklabels(['90','85','75','60','50','60','75','85','90'])
##func=resize_colorbar_vert(cax, axes[1::2], pad=0.0, size=0.02)
##fig.canvas.mpl_connect('draw_event', func)
##cax = fig.add_axes([0.925,0.11,0.0175,0.77])
#
##cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax ,extend='both',orientation='vertical')
##cbar.ax.set_yticklabels(['90','85','75','60','75','85','90'])
#plt.subplots_adjust(left=0.1,bottom=None,top=None,right=None,hspace=0,wspace=0)
#fig.savefig('/home/ascherrmann/009-ERA-5/MED/' + 'cyclones-colored-contribution-new-adv-all-%01dh-2-%d-color.png'%(hcyc,rdis),dpi=300,bbox_inches="tight")
plt.close('all')


