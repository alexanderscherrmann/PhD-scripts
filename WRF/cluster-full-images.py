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
import netCDF4

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


f = open(ps + 'DJF-individual-fields.txt','rb')
data = pickle.load(f)
f.close()


sel = pd.read_csv(ps + 'DJF-intense-cyclones.csv')
sea = 'DJF'
cfl = np.arange(20,90.1,20)
plotvar = ['TH850','U300hPa','PV300hPa','MSL','THE850','Q850','Udiff','THdiff']

cmap_pv,pv_levels,norm_pv,ticklabels_pv=PV_cmap2()

units = ['K',
        'm s$^{-1}$',
        'PVU',
        'hPa',
        'K',
        'kg kg$^{-1}$',
        'm s$^{-1}$',
        'K'
        ]

cmaps = [matplotlib.cm.seismic,
        matplotlib.cm.gnuplot2,
        cmap_pv,
        matplotlib.cm.cividis,
        matplotlib.cm.seismic,
        matplotlib.cm.YlGnBu,
        matplotlib.cm.BrBG,
        matplotlib.cm.RdBu.reversed(),
        ]

levels = [
        np.arange(260,325,3),
        np.arange(-10,50.1,5),
        pv_levels,
        np.arange(990,1031,5),
        np.arange(260,325,3),
        np.arange(0,0.011,0.001),
        np.arange(-12,15,3),
        np.arange(-5,6,1),
        ]

names = ['TH-850hPa','U-300hPa','PV-300hPa','MSL','THE-850hPa','Q-850hPa','U-diff-300hPa','TH-diff-850hPa']


dw = netCDF4.Dataset('/atmosdyn2/ascherrmann/013-WRF-sim/' + sea + '-clim/wrfout_d01_2000-12-01_00:00:00')
uw = wrf.getvar(dw,'U',meta=False)
thw = wrf.getvar(dw,'th',meta=False)
uw = uw[:,:,1:]/2 + uw[:,:,:-1]/2
pw = wrf.getvar(dw,'pressure',meta=False)
uw300 = intp(uw,pw,300,meta=False)
thw850 = intp(thw,pw,850,meta=False)
low = np.linspace(-120,79.5,400)
law = np.linspace(10,79.5,140)

pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/season/DJF/cluster/'
if not os.path.isdir(pi):
    os.mkdir(pi)

for un,lvl,cmap,var,nam in zip(units,levels,cmaps,plotvar,names):
   print(var)
   if var=='THdiff':
    nor = BoundaryNorm(lvl,cmap.N)
    if var=='PV300hPa':
        nor = norm_pv

    if not os.path.isdir(pi + var + '/'):
        os.mkdir(pi + var)

    for q,wi in enumerate(which):
      for ll in [200]:
        for wq,we in zip(wl,when):

            fig = plt.figure(figsize=(19,20))
            gs = gridspec.GridSpec(nrows=5, ncols=3)
            axes = []
            for k in range(5):
                for l in range(3):
                    axes.append(fig.add_subplot(gs[k,l],projection=ccrs.PlateCarree()))
    
            ### avplotfield
            if var=='Udiff' or var=='THdiff':
                avplots = np.zeros((140,400))
            else:
                avplots = np.zeros((141,401)) 
            avcycmask = np.zeros((141,401))
            avwcbout = np.zeros((141,401))
            for ax,label,clus in zip(axes,labels,cluster):
                ### local plot field
                if var=='Udiff' or var=='THdiff':
                    plots=np.zeros((140,400))
                    cycmask0 = np.zeros((140,400))
                else:
                    plots=np.zeros((141,401))
                cycmask = np.zeros((141,401))
                wcbout = np.zeros((141,401))
                ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
                ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
                ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())


                ids = sel['ID'].values[np.where(sel['region'].values==clus)[0]]
                c = len(ids)
                if var=='Udiff':
                    for i in ids:
                        plots+= (data[sea][wi][ll][we][i]['U300hPa'][:-1,:-1]-uw300)/c
                        avplots+= (data[sea][wi][ll][we][i]['U300hPa'][:-1,:-1]-uw300)

                elif var=='THdiff':
                    for i in ids:
                        plots+= (data[sea][wi][ll][we][i]['TH850'][:-1,:-1]-thw850)/c
                        avplots+= (data[sea][wi][ll][we][i]['TH850'][:-1,:-1]-thw850)
                        cycmask0 +=data[sea][wi][ll]['dates'][i]['cycfreq'][:-1,:-1]/c

                else:
                    for i in ids:
                        plots+= data[sea][wi][ll][we][i][var]/c
                        avplots+= data[sea][wi][ll][we][i][var]
                        cycmask += data[sea][wi][ll][we][i]['cycfreq']/c
                        wcbout += data[sea][wi][ll][we][i]['wcbout400freq']/c
                        avwcbout += data[sea][wi][ll][we][i]['wcbout400freq']
                        avcycmask += data[sea][wi][ll][we][i]['cycfreq']

                if var=='MSL':
                    plots/=100
                
                if var=='Udiff' or var=='THdiff':
                    h = ax.contourf(low,law,plots,cmap=cmap,levels=lvl,norm=nor,extend='both')
                    if var=='THdiff':
                        ax.contour(low,law,cycmask0*100,colors='purple',levels=np.arange(20,101,20),linewidths=1)

                else:
                    h = ax.contourf(LON[lons],LAT[lats],plots,cmap=cmap,levels=lvl,norm=nor,extend='both')

                if var=='MSL' or var=='U300hPa':
                    ax.contour(LON[lons],LAT[lats],cycmask*100,colors='r',levels=np.arange(20,101,20),linewidths=1)

                if var=='PV300hPa':
                    ax.contour(LON[lons],LAT[lats],wcbout*100,colors='white',levels=np.arange(20,101,20),linewidths=1)
                ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
                ax.set_yticks(latticks, crs=ccrs.PlateCarree());
                ax.set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
                ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)

                ax.text(0.025,1.02,label + '-%d'%c,fontsize=8,fontweight='bold',color='k',transform=ax.transAxes)

            if var=='MSL':
                avplots/=100

            avplots/=200
            avcycmask/=200
            avwcbout/=200

            axes[-1].add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
            axes[-1].add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
            axes[-1].set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

            if var=='Udiff' or var=='THdiff':
                h = axes[-1].contourf(low,law,avplots,cmap=cmap,levels=lvl,norm=nor,extend='both')
            else:
                h = axes[-1].contourf(LON[lons],LAT[lats],avplots,cmap=cmap,levels=lvl,norm=nor,extend='both')

            axes[-1].text(0.025,1.02,'composite-200',fontsize=8,fontweight='bold',color='k',transform=axes[-1].transAxes)
            axes[-1].set_xticks(lonticks, crs=ccrs.PlateCarree());
            axes[-1].set_yticks(latticks, crs=ccrs.PlateCarree());
            axes[-1].set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
            axes[-1].set_yticklabels(labels=latticks.astype(int),fontsize=10)
            if var=='MSL' or var=='U300hPa':
                axes[-1].contour(LON[lons],LAT[lats],avcycmask*100,colors='r',levels=np.arange(20,101,20),linewidths=1)
            if var=='PV300hPa':
                axes[-1].contour(LON[lons],LAT[lats],avwcbout*100,colors='white',levels=np.arange(20,101,20),linewidths=1)
#                if var=='U300hPa':
#                    ax.contour(LON[lons],LAT[lats],cycfreq*100,linewidths=1,alpha=1,colors='white',levels=np.arange(20,80.1,10))
#                if var=='TH850':
#                    ax.contour(LON[lons],LAT[lats],wcbo*100,linewidths=1,alpha=1,colors='white',levels=np.arange(20,80.1,10))
#                if var=='MSL':
#                    ax.contour(LON[lons],LAT[lats],plots['PV300hPa'],linewidths=1,alpha=1,colors='red',levels=[1.5])
#                    ax.contour(LON[lons],LAT[lats],cycfreq*100,linewidths=1,alpha=1,colors='white',levels=np.arange(20,80.1,10))
#                if var=='THdiff':
#                    ax.contour(LON[lons],LAT[lats],data[mo]['intense-cyclones.csv'][ll]['dates']['cycfreq']*100,linewidths=1,alpha=1,colors='purple',levels=np.arange(20,80.1,10))

            plt.subplots_adjust(right=0.9,wspace=0.1,hspace=0.03,top=0.6,bottom=0.1)
            cbax = fig.add_axes([0.9, 0.11, 0.01, 0.48])
            cbar=plt.colorbar(h, ticks=lvl,cax=cbax)
    
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.set_xlabel(un,fontsize=10)
            cbar.ax.set_xticklabels(lvl)
             
            name = nam + '-cluster-' + wq +'.png'

            fig.savefig(pi + var + '/' +  name,dpi=300,bbox_inches='tight')
            plt.close('all')

