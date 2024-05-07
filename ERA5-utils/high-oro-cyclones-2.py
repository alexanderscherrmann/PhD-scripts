import numpy as np
import os
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
import pickle
import xarray as xr

from useful_functions import create_lonlat_from_file,resize_colorbar_horz,resize_colorbar_vert
from colormaps import PV_cmap2
from conversions import coord2grid
from conversions import level_to_index_T
import netCDF4 as nc
import load_netcdf
from dypy.small_tools import CrossSection
from dypy.netcdf import read_var_bbox
from dypy.small_tools import interpolate
from dypy.tools.py import print_args,ipython
import netCDF4
import math
import dypy.netcdf as nc

import cartopy
import matplotlib.gridspec as gridspec
import functools
import pandas as pd

CT = 'MED'

pload = '/home/ascherrmann/009-ERA-5/MED/ctraj/use/'
f = open(pload + 'PV-data-dPSP-100-ZB-800-2-400-correct-distance.txt','rb')
data = pickle.load(f)
f.close()

NORO = xr.open_dataset('/home/ascherrmann/scripts/ERA5-utils/NORO')


savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']
var = []

pload = '/home/ascherrmann/009-ERA-5/MED/traj/use/'
for u,x in enumerate(savings):
    f = open(pload[:-4] + x + 'furthersel.txt',"rb")
    var.append(pickle.load(f))
    f.close()

SLP = var[0]
lon = var[1]
lat = var[2]
ID = var[3]
hourstoSLPmin = var[4]
avaID = np.array([])
minSLP = np.array([])
for k in range(len(ID)):
    avaID=np.append(avaID,ID[k][0].astype(int))
    minSLP = np.append(minSLP,SLP[k][abs(hourstoSLPmin[k][0]).astype(int)])


df = pd.read_csv(pload[:-4] + 'pandas-all-data.csv')
df = df.loc[df['ntraj075']>=200]
ndf = pd.DataFrame(columns=df.columns)

ID = df['ID'].values
hourstoSLPmin = df['htminSLP'].values
lon = df['lon'].values
lat = df['lat'].values
minSLP = df['minSLP'].values

clim = np.loadtxt('/home/ascherrmann/009-ERA-5/MED/clim-avPV.txt')

df2 = pd.DataFrame(columns=['PV','count','avPV'],index=['Year','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
df2['avPV'] = np.append(np.mean(clim),clim)

psave = '/home/ascherrmann/009-ERA-5/MED/'

oro = data['oro']
datadi = data['rawdata']

dipv = data['dipv']
rdis = 400
H = 48
a = 1

PVstart = np.array([])
PVend = np.array([])
adv = np.array([])
cyc = np.array([])
env = np.array([])
oroa = np.array([])

minpltlatc = 30
minpltlonc = -5

maxpltlatc = 50
maxpltlonc = 50

soroc = np.array([]) 
soroe = np.array([])
loroe = np.array([])
loroc = np.array([])

c = 'cyc'
e = 'env'

#fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
#ax.coastlines()

fig=plt.figure(figsize=(6,4))
gs = gridspec.GridSpec(ncols=1, nrows=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)

orocounter = 0
sorocounter =0

LON=np.arange(-180,180.1,0.5)
LAT=np.arange(-90,90.1,0.5)
counter = np.zeros((len(LAT),len(LON)))

cycano = np.array([])
envano = np.array([])
advano = np.array([])
advano2 = np.array([])

hcyc = 6
tj = 0
con = 0
slpdis = np.array([])
soroslp = np.array([])
print('ID\tcyc\tenv\toro\ttoro\ttraj\tlon\tlat')
for qq,date in enumerate(dipv.keys()):
    if np.all(ID!=int(date)):
        continue
 #if date=='108215' or date=='269867':
    q = np.where(ID==int(date))[0][0]
    MON = df['mon'].values[q]
#    if date!='540325' and date!='460854':
#        continue

    if (hourstoSLPmin[q]< hcyc):
        continue
    con +=1

    d = date
    idp = np.where(datadi[date]['PV'][:,0]>=0.75)[0]
    cLLon = lon[q]#[abs(hourstoSLPmin[q][0]).astype(int)]
    cLLat = lat[q]#[abs(hourstoSLPmin[q][0]).astype(int)]
    slpdis = np.append(slpdis,SLP[q])#[abs(hourstoSLPmin[q][0]).astype(int)])
#    tralon = datadi[date]['lon'][:,:]
#    tralat = datadi[date]['lat'][:,:]
    PV = datadi[date]['PV'][idp,:]

    pvstart = PV[:,-1]
    pvend = PV[:,0]
    PVstart = np.append(PVstart,pvstart)
    PVend = np.append(PVend,pvend)
    cypv = dipv[d][c][idp,0]

    enpv = dipv[d][e][idp,0]

    adv = np.append(adv,pvstart/pvend)
    cyc = np.append(cyc,cypv/pvend)
    env = np.append(env,enpv/pvend)

    cycano = np.append(cycano,cypv/(pvend-df2['avPV'][MON]))
    envano = np.append(envano,enpv/(pvend-df2['avPV'][MON]))
    advano = np.append(advano,(pvstart-df2['avPV'][MON])/(pvend-df2['avPV'][MON]))
    advano2 = np.append(advano2,pvstart/(pvend-df2['avPV'][MON]))

    PVoro = oro[date]['env'][idp,:]

    pvoro = oro[date]['env'][idp,0]# + oro[date]['cyc'][idp,0]
    if np.where(pvoro!=0)[0].size==0:
        continue
    orot = len(np.where(pvoro!=0)[0])# & ((abs(oro[date]['cyc'][idp,0])<0.05)))[0])
    if len(idp)<tj:
        continue
    if df['envperano'].values[q]>=0.65 and np.mean(pvoro/(pvend-df2['avPV'][MON]))>0.5:
    #((np.mean(pvoro/(pvend-df2['avPV'][MON]))>0.7)):# & (df['envperano']>0.80)):
     orocounter+=1
     if NORO['ZB'][0,np.where(abs(NORO['lat']-cLLat)==np.min(abs(NORO['lat']-cLLat)))[0][0],np.where(abs(NORO['lon']-cLLon)==np.min(abs(NORO['lon']-cLLon)))[0][0]]<1:
        sorocounter+=1
        soroslp = np.append(soroslp,SLP[q])
        soroc = np.append(soroc,np.mean(oro[date]['cyc'][idp,0]/(pvend-df2['avPV'][MON])))
        soroe = np.append(soroe,np.mean(oro[date]['env'][idp,0]/(pvend-df2['avPV'][MON])))

        oroa = np.append(oroa,pvoro/(pvend-df2['avPV'][MON]))
     else:
         loroc = np.append(loroc,np.mean(oro[date]['cyc'][idp,0]/pvend))
         loroe = np.append(loroe,np.mean(oro[date]['env'][idp,0]/pvend))

     if lat[q]<30 or lat[q]>48:
         continue
     if lon[q]<-5 or lon[q]>42:
         continue
     if lon[q]<2 and lat[q]>42:
         continue

     if (lon[q]%0.5!=0):
         lon[q]=np.round(lon[q],0)
     if (lat[q]%0.5!=0):
         lat[q]=np.round(lat[q],0)
     if lat[q]<30:
         continue
     lo = np.where(np.round(LON,1)==np.round(lon[q],1))[0][0]
     la = np.where(np.round(LAT,1)==np.round(lat[q],1))[0][0]
     counter[la,lo]+=1
     ndf = ndf.append(df.loc[df['ID']==int(date)])

ax.plot([],[],color='k',ls=' ',marker='.',markersize=1)
ax.plot([],[],color='k',ls=' ',marker='.',markersize=np.max(counter))
lati,loni = np.where(counter!=0)
#ax.scatter(LON[loni],LAT[lati],marker='.',color='k',s=counter[lati,loni]*10,zorder=2)
#ax.legend(['1 cyclone','%d cyclones'%np.max(counter)],loc='upper left',fontsize=8)
ax.scatter(ndf['lon'].values,ndf['lat'].values,color='k',s=1)
for k in range(5,len(ndf['lon'].values),6):
    ax.annotate('%d'%ndf['ID'].values[k],(ndf['lon'].values[k],ndf['lat'].values[k]),fontsize=6)
    print(ndf['ID'].values[k])
#        if np.mean(datadi[d]['OL'][idp,0]<0.7):
#        ax.text(cLLon-1,cLLat,date + '-%d'%len(idp),fontsize=4)
#        print('%d\t%.2f\t%.2f\t%.2f\t%.2f\t%d\t%.1f\t%.1f'%(int(d),np.mean(cypv/pvend),np.mean(enpv/pvend),np.mean(pvoro/pvend),orot/len(idp),len(idp),cLLon,cLLat))

print(orocounter,sorocounter)
print(np.sum(counter))
print(con)

print(np.mean(ndf['envper'].values))
NORO = xr.open_dataset('/home/ascherrmann/scripts/ERA5-utils/NORO')
ZB = NORO['ZB'].values[0]
Zlon = NORO['lon']
Zlat = NORO['lat']
Emaxv = 3000
Eminv = 800
elevation_levels = np.arange(Eminv,Emaxv,400)

ax.contour(Zlon,Zlat,ZB,levels = elevation_levels,colors='purple',linewidths=0.35,alpha=1)
lonticks=np.arange(minpltlonc, maxpltlonc+1,5)
latticks=np.arange(minpltlatc, maxpltlatc+1,5)
ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
ax.set_yticks(latticks, crs=ccrs.PlateCarree());
ax.set_xticklabels(labels=lonticks,fontsize=8)
ax.set_yticklabels(labels=latticks,fontsize=8)
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_extent([minpltlonc, maxpltlonc-5, minpltlatc, maxpltlatc], ccrs.PlateCarree())
#ax.text(0.03, 0.95, 'b)', transform=ax.transAxes,fontsize=12, fontweight='bold',va='top')
figname = psave + 'high-oro-mature-sel-anomaly-%dh.png'%hcyc
fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close()




flier = dict(marker='+',markerfacecolor='grey',markersize=1,linestyle=' ',markeredgecolor='grey')
meanline = dict(linestyle='-',linewidth=1,color='red')
#
meanline2 = dict(linestyle='-',linewidth=1,color='dodgerblue')
capprops = dict(linestyle='-',linewidth=1,color='grey')
medianprops = dict(linestyle='-',linewidth=1,color='grey')
medianprops1 = dict(linestyle='-',linewidth=1,color='black')
boxprops = dict(linestyle='-',linewidth=1.,color='slategrey')
whiskerprops= dict(linestyle='-',linewidth=1,color='grey')

boxdata = [np.sort(loroc)*100,np.sort(soroc)*100,np.sort(loroe)*100,np.sort(soroe)*100]
labs = ['land cyc.','sea cyc.','land env.','sea env.']

fig,ax = plt.subplots()
ax.set_ylabel(r'contribution to PVend [%]')
ax.set_ylim(-25,100)
ax.set_xlim(0,5)
bp = ax.boxplot(boxdata,whis=(10,90),labels=labs,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops1)
ax.tick_params(labelright=False,right=True)
ax.set_xticklabels(labels=labs)
#fig.savefig('/home/ascherrmann/009-ERA-5/MED/boxwis-oroPV-t-%d-tr-%d.png'%(hcyc,tj),dpi=300,bbox_inches="tight")
plt.close('all')

