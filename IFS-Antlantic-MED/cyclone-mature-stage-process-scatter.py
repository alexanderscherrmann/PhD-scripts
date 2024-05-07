import numpy as np
import pickle

CT = 'MED'
pload ='/home/ascherrmann/010-IFS/traj/MED/use/'

f = open(pload + 'PV-data-MEDdPSP-100-ZB-800PVedge-0.3.txt','rb')
data = pickle.load(f)
f.close()
datadi = data['rawdata']
dipv = data['dipv']
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

def colbar(cmap,minval,maxval,nlevels):
    maplist = [cmap(i) for i in range(cmap.N)]
    newmap = ListedColormap(maplist)
    lv = np.linspace(minval,maxval,nlevels)
    norm = BoundaryNorm(lv,cmap.N)
    return newmap, norm

LON=np.linspace(-180,180,901)
LAT=np.linspace(0,90,226)

if CT=='MED':
    minpltlonc = -5
    maxpltlonc = 45
    minpltlatc = 30
    maxpltlatc = 50
    steps = 5
lonticks=np.arange(minpltlonc+5, maxpltlonc+1,steps*2)
latticks=np.arange(minpltlatc, maxpltlatc+1,steps)

apl = cm.seismic
minv = -1.0 #* 100
mav = 1.0 #* 100
steps = 0.25 #* 100
lvls = np.arange(minv,(mav+0.000001),steps)
sp = np.linspace(-1.,1.,256)
labs = helper.traced_vars_IFS()
cmap,norm = colbar(apl,minv,mav,len(lvls))
cmap.set_under('black')
cmap.set_over('darkorange')
#for  k in range(48,80):
for k in range(96,160):
    cmap.colors[k] = np.array([189/256, 195/256, 199/256, 1.0])
#print(cmap(1))
#print(cmap(100))

labels = ['a)','b)','c)','d)','e)','f)']

proc = ['PVRCONVT','PVRTURBT','PVRCONVM','PVRTURBM','PVRLS','APVRAD']#'PVRLWH','PVRLWC']
lab = ['CONVT','TURBT','CONVM','TURBM','LS','RAD']#LWH','LWC']
### use CONVT in first and TURBT in second PVR-T
for qq, key in enumerate(['cyc','env']):
 fig, axes = plt.subplots(3, 2, subplot_kw=dict(projection=ccrs.PlateCarree()),sharex=True,sharey=True)
 axes = axes.flatten()

 for q, ax in enumerate(axes[:]):
    ax.coastlines()
    for ul, date in enumerate(datadi.keys()):
        idp = np.where(datadi[date]['PV'][:,0]>=0.75)[0]
        lon = np.mean(datadi[date]['lon'][:,0])
        lat = np.mean(datadi[date]['lat'][:,0])

        colval = np.mean(dipv[date][key][proc[q]][idp,0])#/np.mean(dipv[date][key]['APVTOT'][idp,0])
        
#        if q==0:
#            if abs(np.sum(dipv[date][key]['PVRCONVT'][idp,0]))<abs(np.sum(dipv[date][key]['PVRTURBT'][idp,0])):
#                colval = 0
#        if q==1:
#            if abs(np.sum(dipv[date][key]['PVRCONVT'][idp,0]))>abs(np.sum(dipv[date][key]['PVRTURBT'][idp,0])):
#                colval = 0
        ids = np.where(abs(sp-colval)==np.min(abs(sp-colval)))[0][0]
        if ids==255:
            ids+=1
        if ids==0:
            ids-=1

        col = cmap(ids)
        ac = ax.scatter(lon,lat,color=col,marker='o',zorder=1,s=3)
    ax.set_xticks(lonticks, crs=ccrs.PlateCarree())
    ax.set_yticks(latticks, crs=ccrs.PlateCarree())

    if q%2==0:
        ax.set_yticklabels(labels=latticks,fontsize=10)
        ax.yaxis.set_major_formatter(LatitudeFormatter())
    if q==4 or q==5:
        ax.set_xticklabels(labels=lonticks,fontsize=10)
        ax.xaxis.set_major_formatter(LongitudeFormatter())

    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc])#, ccrs.PlateCarree())
    ax.text(0.06, 0.85, labels[q], transform=ax.transAxes,fontsize=12, fontweight='bold',va='top')
    ax.text(0.45, 0.95, lab[q], transform=ax.transAxes,fontsize=8,va='top')

 cax = fig.add_axes([0.925,0.11,0.0175,0.77])
 cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax ,extend='both',orientation='vertical')
 cbar.ax.set_xlabel('PVU')
 for t in cbar.ax.get_yticklabels():
     t.set_fontsize(8)
 plt.subplots_adjust(left=0.1,bottom=None,top=None,right=0.925,hspace=0,wspace=0)
 fig.savefig('/home/ascherrmann/010-IFS/' + 'cyclones-processes-scatter-full-convt-turbt-' + key + '.png',dpi=300,bbox_inches="tight")
 plt.close('all')


minv = -1.0 * 100
mav = 1.0 * 100
steps = 0.25 * 100
lvls = np.arange(minv,(mav+0.000001),steps)
sp = np.linspace(-100.,100.,256)

cmap,norm = colbar(apl,minv,mav,len(lvls))
cmap.set_under('black')
cmap.set_over('darkorange')
#for  k in range(48,80):
for k in range(96,160):
    cmap.colors[k] = np.array([189/256, 195/256, 199/256, 1.0]) 


for qq, key in enumerate(['cyc','env']):
 fig, axes = plt.subplots(3, 2, subplot_kw=dict(projection=ccrs.PlateCarree()),sharex=True,sharey=True)
 axes = axes.flatten()

 for q, ax in enumerate(axes[:]):
    ax.coastlines()
    for ul, date in enumerate(datadi.keys()):
        idp = np.where(datadi[date]['PV'][:,0]>=0.75)[0]
        lon = np.mean(datadi[date]['lon'][:,0])
        lat = np.mean(datadi[date]['lat'][:,0])

        colval = np.mean(dipv[date][key][proc[q]][idp,0])/np.mean(dipv[date][key]['APVTOT'][idp,0]) * 100
        if q==0:
            if abs(np.sum(dipv[date][key]['PVRCONVT'][idp,0]))<abs(np.sum(dipv[date][key]['PVRTURBT'][idp,0])):
                colval = 0
        if q==1:
            if abs(np.sum(dipv[date][key]['PVRCONVT'][idp,0]))>abs(np.sum(dipv[date][key]['PVRTURBT'][idp,0])):
                colval = 0
        ids = np.where(abs(sp-colval)==np.min(abs(sp-colval)))[0][0]

        if ids==255:
            ids+=1
        if ids==0:
            ids-=1
        col = cmap(ids)
        ac = ax.scatter(lon,lat,color=col,marker='o',zorder=1,s=3)
    ax.set_xticks(lonticks, crs=ccrs.PlateCarree())
    ax.set_yticks(latticks, crs=ccrs.PlateCarree())

    if q%2==0:
        ax.set_yticklabels(labels=latticks,fontsize=10)
        ax.yaxis.set_major_formatter(LatitudeFormatter())
    if q==4 or q==5:
        ax.set_xticklabels(labels=lonticks,fontsize=10)
        ax.xaxis.set_major_formatter(LongitudeFormatter())

    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc])#, ccrs.PlateCarree())
    ax.text(0.06, 0.85, labels[q], transform=ax.transAxes,fontsize=12, fontweight='bold',va='top')
    ax.text(0.45, 0.95, lab[q], transform=ax.transAxes,fontsize=8,va='top')

 cax = fig.add_axes([0.925,0.11,0.0175,0.77])
 cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax ,extend='both',orientation='vertical')
 cbar.ax.set_xlabel('%')
 for t in cbar.ax.get_yticklabels():
     t.set_fontsize(8)
 plt.subplots_adjust(left=0.1,bottom=None,top=None,right=0.925,hspace=0,wspace=0)
 fig.savefig('/home/ascherrmann/010-IFS/' + 'cyclones-processes-scatter-percentage-' + key + '.png',dpi=300,bbox_inches="tight")
 plt.close('all')

minv = -0.6
mav = 0.6
steps = 0.15
lvls = np.arange(minv,(mav+0.000001),steps)
sp = np.linspace(-0.6,0.6,256)

cmap,norm = colbar(apl,minv,mav,len(lvls))
cmap.set_under('black')
cmap.set_over('darkorange')
for k in range(96,160):
    cmap.colors[k] = np.array([189/256, 195/256, 199/256, 1.0])

fig,ax = plt.subplots(1,1,subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.coastlines()
for q,date in enumerate(data['rawdata'].keys()):
    PVf = data['rawdata'][date]['PV']
    lon = np.mean(datadi[date]['lon'][:,0])
    lat = np.mean(datadi[date]['lat'][:,0])
    datadi = data['rawdata']
    idp = np.where(PVf[:,0]>=0.75)[0]

    datadi[date]['DELTAPV'] = np.zeros(datadi[date]['PV'].shape)
    datadi[date]['DELTAPV'][:,1:] = datadi[date]['PV'][:,:-1]-datadi[date]['PV'][:,1:]
    datadi[date]['RES'] = np.zeros(datadi[date]['PV'].shape)
    datadi[date]['PVRTOT'] = np.zeros(datadi[date]['PV'].shape)
    for pv in labs[8:]:
        datadi[date]['PVRTOT']+=datadi[date][pv]

    datadi[date]['RES'][:,:-1] = np.flip(np.cumsum(np.flip(datadi[date]['PVRTOT'][:,1:],axis=1),axis=1),axis=1)-np.flip(np.cumsum(np.flip(datadi[date]['DELTAPV'][:,1:],axis=1),axis=1),axis=1)
    RES = np.mean(datadi[date]['RES'][idp,0])
    colval = RES
    ids = np.where(abs(sp-colval)==np.min(abs(sp-colval)))[0][0]
    if ids==255:
            ids+=1
    if ids==0:
            ids-=1
    col = cmap(ids)
    ac = ax.scatter(lon,lat,color=col,marker='o',zorder=1,s=8)
ax.set_xticks(lonticks, crs=ccrs.PlateCarree())
ax.set_yticks(latticks, crs=ccrs.PlateCarree())

ax.set_yticklabels(labels=latticks,fontsize=10)
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_xticklabels(labels=lonticks,fontsize=10)
ax.xaxis.set_major_formatter(LongitudeFormatter())

ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc])#, ccrs.PlateCarree())

#cax = fig.add_axes([0.925,0.11,0.0175,0.77])
cax = fig.add_axes([0, 0, 0.1, 0.1])
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax ,extend='both',orientation='vertical')
func=resize_colorbar_vert(cax, ax, pad=0.01, size=0.02)
fig.canvas.mpl_connect('draw_event', func)
cbar.ax.set_xlabel('PVU')
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(8)
fig.savefig('/home/ascherrmann/010-IFS/' + 'cyclones-residuals.png',dpi=300,bbox_inches="tight")
plt.close('all')



