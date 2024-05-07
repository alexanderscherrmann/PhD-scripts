import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import wrf
from wrf import interplevel as intp
from netCDF4 import Dataset as ds
import os
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec
from wrf import interplevel as intp
import netCDF4
import glob
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert

import pickle

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'

when = ['fourdaypriormature','fivedaypriormature','sixdaypriormature','sevendaypriormature','threedaypriormature','twodaypriormature','onedaypriormature','dates']

cluster = ['balearic','adriaticN','adriaticS','ionian-aegean','sicily','cyprus','black','africaE','africaC','tyrric','genua','centralMed','greece','belowgreece']
labels = ['Balearic','AdriaticN','AdriaticS','Ionian-Aegean','Sicily','Cyprus','Black','AfricaE','AfricaC','Tyrric','Genoa','Central','Greece','Below greece']

wl = ['4','5','6','7','3','2','1','0']

which = ['intense-cyclones.csv']
minlon = -120
minlat = 10
maxlat = 80
maxlon = 80

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)
lonshort = np.linspace(-120,80,401)
latshort = np.linspace(10,80,141)

lonticks=np.arange(minlon, maxlon+1,20)
latticks=np.arange(minlat, maxlat+1,10)

lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

# average cyclone 


sel = pd.read_csv(ps + 'DJF-intense-cyclones.csv')
ps += 'PV-med-traj/'
sea = 'DJF'
plotvar = ['TH','PV','P']

cmap_pv,pv_levels,norm_pv,ticklabels_pv=PV_cmap2()

units = ['K',
        'PVU',
        'hPa',
        ]

cmaps = [matplotlib.cm.seismic,
        cmap_pv,
        matplotlib.cm.jet
        ]

levels = [
        np.arange(290,340,3),
        pv_levels,
        np.arange(200,400,10),
        ]

names = ['TH-evol','PV-evol','origin-scatter']
H = 192
HH = 24

pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/season/DJF/cluster/'
if not os.path.isdir(pi):
    os.mkdir(pi)


for un,lvl,cmap,var,nam in zip(units,levels,cmaps,plotvar,names):
#  if var=='TH':
    nor = BoundaryNorm(lvl,cmap.N)
    if var=='PV300hPa':
        nor = norm_pv

    if not os.path.isdir(pi + var + '/'):
        os.mkdir(pi + var)

    for q,wi in enumerate(which):
        if var=='P':
          figfr = plt.figure(figsize=(19,20))
          gsfr = gridspec.GridSpec(nrows=5, ncols=3)
          axesfr = []

          fig = plt.figure(figsize=(19,20))
          gs = gridspec.GridSpec(nrows=5, ncols=3)
          axes = []
          for k in range(5):
              for l in range(3):
                  axes.append(fig.add_subplot(gs[k,l],projection=ccrs.PlateCarree()))
                  axesfr.append(figfr.add_subplot(gsfr[k,l],projection=ccrs.PlateCarree()))

    
          ### avplotfield
          for axfr,ax,label,clus in zip(axesfr,axes,labels,cluster):
              ### local plot field
              ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
              ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
              ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

              axfr.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
              axfr.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
              axfr.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

              tf = sel.iloc[sel['region'].values==clus]
              freq = np.zeros((361,721))
              freqc = 0
              for d in tf['dates'].values:
                  c = glob.glob(ps + '*' + d + '*.txt')
                  traj = np.loadtxt(c[0],skiprows=5)
                  lon = traj[:,1].reshape(-1,H+1)
                  lat = traj[:,2].reshape(-1,H+1)
                  P = traj[:,3].reshape(-1,H+1)
                  for la,lo in zip(lat[:,HH],lon[:,HH]):
                      loi = np.where(abs(LON-lo)==np.min(abs(LON-lo)))[0][0]
                      lai = np.where(abs(LAT-la)==np.min(abs(LAT-la)))[0][0]
                      freq[lai,loi]+=1
                      freqc+=1

                  for ma,we in zip(['s'],[144]):
#                  for ma,we in zip(['.','+','d','o','+','s'],np.arange(24,145,24)):
                      ax.scatter(lon[:,we],lat[:,we],marker=ma,c=P[:,we],cmap=cmap,norm=BoundaryNorm(lvl,cmap.N))
                      axes[-1].scatter(lon[:,we],lat[:,we],marker=ma,c=P[:,we],cmap=cmap,norm=BoundaryNorm(lvl,cmap.N))

              freq/=freqc
              c = len(tf['ID'].values)
              axfr.contourf(LON,LAT,freq,levels=np.array([1e-6,1e-4,1e-2,5e-2]),colors='r',linewidths=1)
              axfr.set_xticks(lonticks, crs=ccrs.PlateCarree());
              axfr.set_yticks(latticks, crs=ccrs.PlateCarree());
              axfr.set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
              axfr.set_yticklabels(labels=latticks.astype(int),fontsize=10)
              axfr.text(0.025,1.02,label + '-%d'%c,fontsize=8,fontweight='bold',color='k',transform=axfr.transAxes)
              try:
                  figfr.subplots_adjust(right=0.9,wspace=0.1,hspace=0.03,top=0.6,bottom=0.1)
              except:
                  print('didnot work')
              ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
              ax.set_yticks(latticks, crs=ccrs.PlateCarree());
              ax.set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
              ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)

              ax.text(0.025,1.02,label + '-%d'%c,fontsize=8,fontweight='bold',color='k',transform=ax.transAxes)

          axes[-1].add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
          axes[-1].add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
          axes[-1].set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

          axes[-1].text(0.025,1.02,'composite-200',fontsize=8,fontweight='bold',color='k',transform=axes[-1].transAxes)
          axes[-1].set_xticks(lonticks, crs=ccrs.PlateCarree());
          axes[-1].set_yticks(latticks, crs=ccrs.PlateCarree());
          axes[-1].set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
          axes[-1].set_yticklabels(labels=latticks.astype(int),fontsize=10)

          plt.subplots_adjust(right=0.9,wspace=0.1,hspace=0.03,top=0.6,bottom=0.1)
#          cbax = fig.add_axes([0.9, 0.11, 0.01, 0.48])
#          cbar=plt.colorbar(h, ticks=lvl,cax=cbax)
    
#          cbar.ax.tick_params(labelsize=10)
#          cbar.ax.set_xlabel(un,fontsize=10)
#          cbar.ax.set_xticklabels(lvl)
           
          name = nam + '-cluster.png'

          fig.savefig(pi + var + '/' +  name,dpi=300,bbox_inches='tight')
          name = nam + 'freq-cluster.png'
          figfr.savefig(pi + var + '/' +  name,dpi=300,bbox_inches='tight')
          plt.close('all')
        elif var=='TH':    
          fig,axes = plt.subplots(5,3,sharex=True,sharey=True)
          axes=axes.flatten()
          avpv = np.zeros(HH+1)
          avth = np.zeros(HH+1)
          avc = 0
          for qk,ax,label,clus in zip(range(len(axes)),axes,labels,cluster):
              tf = sel.iloc[sel['region'].values==clus]
              axpv = np.zeros(HH+1)
              axth = np.zeros(HH+1)
              av = 0
              for d in tf['dates'].values:
                  c = glob.glob(ps + '*' + d + '*.txt')
                  traj = np.loadtxt(c[0],skiprows=5)
                  ti = traj[:,0].reshape(-1,H+1)[0,:HH+1]
                  PV = traj[:,-2].reshape(-1,H+1)
                  TH = traj[:,-1].reshape(-1,H+1)

                  axpv += np.sum(PV[:,:HH+1],axis=0)
                  avpv += np.sum(PV[:,:HH+1],axis=0)
                  axth += np.sum(TH[:,:HH+1],axis=0)
                  avth += np.sum(TH[:,:HH+1],axis=0)
                  avc += len(PV[:,0])
                  av += len(PV[:,0])

              axpv/=av
              axth/=av
                  
              ax2 = ax.twinx()
              ax.plot(ti,axpv,color='k')
              ax.set_ylim(4,5.5)
              ax2.plot(ti,axth,color='grey')
              ax2.set_ylim(315,325)
              ax2.tick_params(right=False,labelright=False)
              ax.set_xticks(ticks=np.arange(-120,1,48))
              ax.text(0.4,0.9,clus,transform=ax.transAxes,fontweight='bold',fontsize=6)
              ax.set_xlim(-144,0)
              if qk%3==0:
                  ax.set_ylabel('PV [PVU]')
              if qk%3==2:
                  ax2.set_yticks(ticks=np.array([315,320,325]))
                  ax2.set_yticklabels(labels=np.array([315,320,325]))
                  ax2.set_ylabel('TH [K]')
              if qk>12:
                  ax.set_xlabel('time [h]')
              
          ax = axes[-1]
          ax2 = ax.twinx()
          ax.set_xticks(ticks=np.arange(-120,1,48))
          ax.plot(ti,axpv,color='k')
          ax.set_ylim(3,6)

          ax2.plot(ti,axth,color='grey')
          ax2.set_ylabel('TH [K]')
          ax2.set_ylim(310,325)
          ax.set_xlabel('time [h]')
          plt.subplots_adjust(wspace=0,hspace=0)

          name = 'avPV-TH-evolution.png'
          fig.savefig(pi + name, dpi=300,bbox_inches='tight')
          plt.close('all')



