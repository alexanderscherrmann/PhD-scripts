import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from wrf import interplevel as intp
from netCDF4 import Dataset as ds
import os
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec

import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert

import pickle

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
era5 = '/atmosdyn2/era5/cdf/'

when = ['fourdaypriormature','fivedaypriormature','sixdaypriormature','sevendaypriormature','threedaypriormature','twodaypriormature','onedaypriormature','dates']

wl = ['4','5','6','7','3','2','1','0']

which = ['weak-cyclones.csv','moderate-cyclones.csv','intense-cyclones.csv']
minlon = -120
minlat = 10
maxlat = 80
maxlon = 80

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)
lonshort = np.linspace(-120,80,401)
latshort = np.linspace(10,80,141)
lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

# average cyclone 


f = open(ps + 'monthly-average-fields.txt','rb')
data = pickle.load(f)
f.close()

cfl = np.arange(20,90.1,20)
months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
monthsn = np.arange(1,13)
plotvar = ['TH850','U300hPa','omega500','PV300hPa','Precip24','MSL','wcboutdiff','cycfreqdiff','wcbascdiff','Udiff','PVdiff','THdiff','Precipdiff','THE850','Pat1.5PVU','Pat2PVU','Q850','THEdiff','Qdiff']

cmap_pv,pv_levels,norm_pv,ticklabels_pv=PV_cmap2()

units = ['K','m s$^{-1}$','Pa s$^{-1}$','PVU','mm','hPa',r'%',r'%',r'%','m s$^{-1}$','PVU','K','mm','K','hPa','hPa','kg kg$^{-1}$','K','kg kg$^{-1}$']
cmaps = [matplotlib.cm.seismic,matplotlib.cm.gnuplot2,matplotlib.cm.BrBG,cmap_pv,matplotlib.cm.YlGnBu,matplotlib.cm.cividis,matplotlib.cm.BrBG,matplotlib.cm.BrBG,matplotlib.cm.BrBG,matplotlib.cm.PiYG,matplotlib.cm.PiYG,matplotlib.cm.RdBu.reversed(),matplotlib.cm.PuOr,matplotlib.cm.seismic,matplotlib.cm.jet,matplotlib.cm.jet,matplotlib.cm.YlGnBu,matplotlib.cm.RdBu.reversed(),matplotlib.cm.BrBG]
levels = [np.arange(260,325,3),np.arange(-10,50.1,5),np.arange(-0.3,0.4,0.1),pv_levels,np.arange(0,11,1),np.arange(990,1031,5),np.arange(-30,35,5),np.arange(-30,35,5),np.arange(-30,35,5),np.arange(-15,16,3),np.arange(-1,1.1,0.2),np.arange(-5,6,1),np.arange(-5,6,1),np.arange(260,325,3),np.arange(200,501,25),np.arange(200,501,25),np.arange(0,0.011,0.001),np.arange(-5,6,1),np.arange(-0.0025,0.0026,0.0005)]
names = ['TH-850hPa','U-300hPa','omega-500hPa','PV-300hPa','Precip24','MSL','wcbout-diff','cycfreq-diff','wcbasc-diff','U300-diff','PV300-diff','TH850-diff','Precip24-diff','THE-850hPa','P-at-1.5PVU','P-at-2.0PVU','Q-850hPa','THE850-diff','Q850-diff']

pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/monthly/'
if not os.path.isdir(pi):
    os.mkdir(pi)

for un,lvl,cmap,var,nam in zip(units,levels,cmaps,plotvar,names):
#  if var=='PV300hPa' or var=='Udiff' or var=='PVdiff':
#  if var=='Precipdiff':
   if var=='THdiff':
    nor = BoundaryNorm(lvl,cmap.N)
    if var=='PV300hPa':
        nor = norm_pv

    if not os.path.isdir(pi + var + '/'):
        os.mkdir(pi + var)

    for q,wi in enumerate(which):
      if var[-1]=='f' and q>0:
            continue

      for ll in [50]:
        ### calc average
        plots = dict()
        for wq,we in zip(wl,when):
            fig = plt.figure(figsize=(19,16))
            gs = gridspec.GridSpec(nrows=4, ncols=3)
            axes = []
            for k in range(4):
                for l in range(3):
                    axes.append(fig.add_subplot(gs[k,l],projection=ccrs.PlateCarree()))

            for ax,mo in zip(axes,months):
                ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
                ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
                ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

                
                plots['TH850'] = data[mo][wi][ll][we]['TH850']
                plots['THE850'] = data[mo][wi][ll][we]['THE850']
                plots['Pat1.5PVU'] =data[mo][wi][ll][we]['Pat1.5PVU']
                plots['U300hPa'] = data[mo][wi][ll][we]['U300hPa']
                plots['PV300hPa'] = data[mo][wi][ll][we]['PV300hPa']
                LSP24 = data[mo][wi][ll][we]['lprecip-24h']
                CP24 = data[mo][wi][ll][we]['cprecip-24h']
                plots['omega500'] = data[mo][wi][ll][we]['omega500']
                cycfreq = data[mo][wi][ll][we]['cycfreq']
                SLPcycfreq = data[mo][wi][ll][we]['SLPcycfreq']
                wcba = data[mo][wi][ll][we]['wcbascfreq']
                wcbo = data[mo][wi][ll][we]['wcbout400freq']
                plots['MSL'] = data[mo][wi][ll][we]['MSL']/100
                plots['Precip24'] = (LSP24 + CP24)*1000
                plots['Q850'] = data[mo][wi][ll][we]['Q850']

                plots['wcboutdiff'] = (data[mo]['intense-cyclones.csv'][50][we]['wcbout400freq'] - data[mo]['weak-cyclones.csv'][50][we]['wcbout400freq'])*100
                plots['cycfreqdiff'] = (data[mo]['intense-cyclones.csv'][50][we]['cycfreq'] - data[mo]['weak-cyclones.csv'][50][we]['cycfreq'])*100
                plots['wcbascdiff'] = (data[mo]['intense-cyclones.csv'][50][we]['wcbascfreq'] - data[mo]['weak-cyclones.csv'][50][we]['wcbascfreq'])*100

                plots['Udiff'] = data[mo]['intense-cyclones.csv'][50][we]['U300hPa'] - data[mo]['weak-cyclones.csv'][50][we]['U300hPa']
                plots['PVdiff'] = data[mo]['intense-cyclones.csv'][50][we]['PV300hPa'] - data[mo]['weak-cyclones.csv'][50][we]['PV300hPa']
                plots['THdiff'] = data[mo]['intense-cyclones.csv'][50][we]['TH850'] - data[mo]['weak-cyclones.csv'][50][we]['TH850']
                plots['Precipdiff'] = (data[mo]['intense-cyclones.csv'][50][we]['lprecip-24h'] - data[mo]['weak-cyclones.csv'][50][we]['lprecip-24h'] + data[mo]['intense-cyclones.csv'][50][we]['cprecip-24h'] - data[mo]['weak-cyclones.csv'][50][we]['cprecip-24h'])*1000
                plots['THEdiff'] = data[mo]['intense-cyclones.csv'][50][we]['THE850'] - data[mo]['weak-cyclones.csv'][50][we]['THE850']
                plots['Qdiff'] = data[mo]['intense-cyclones.csv'][50][we]['Q850'] - data[mo]['weak-cyclones.csv'][50][we]['Q850']


            
                h = ax.contourf(LON[lons],LAT[lats],plots[var],cmap=cmap,levels=lvl,norm=nor,extend='both')
                if var=='U300hPa':
                    ax.contour(LON[lons],LAT[lats],cycfreq*100,linewidths=1,alpha=1,colors='white',levels=np.arange(20,80.1,10))
                if var=='TH850':
                    ax.contour(LON[lons],LAT[lats],wcbo*100,linewidths=1,alpha=1,colors='white',levels=np.arange(20,80.1,10))
                if var=='MSL':
                    ax.contour(LON[lons],LAT[lats],plots['PV300hPa'],linewidths=1,alpha=1,colors='red',levels=[1.5])
                    ax.contour(LON[lons],LAT[lats],cycfreq*100,linewidths=1,alpha=1,colors='white',levels=np.arange(20,80.1,10))
                if var=='THdiff':
                    ax.contour(LON[lons],LAT[lats],data[mo]['intense-cyclones.csv'][ll]['dates']['cycfreq']*100,linewidths=1,alpha=1,colors='purple',levels=np.arange(20,80.1,10))

                lonticks=np.arange(minlon, maxlon+1,20)
                latticks=np.arange(minlat, maxlat+1,10)
    
                ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
                ax.set_yticks(latticks, crs=ccrs.PlateCarree());
                ax.set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
                ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)

                ax.text(0.025,1.02,mo,fontsize=8,fontweight='bold',color='k',transform=ax.transAxes)
    
            plt.subplots_adjust(right=0.9,wspace=0.1,hspace=0.03,top=0.6,bottom=0.1)
            cbax = fig.add_axes([0.9, 0.11, 0.01, 0.48])
            cbar=plt.colorbar(h, ticks=lvl,cax=cbax)
    
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.set_xlabel(un,fontsize=10)
            cbar.ax.set_xticklabels(lvl)
             
            name = nam + '-for-' + wi[:-4] + '-' + wq + '-monthly-' + '%d'%ll +'.png'
            if var[-1]=='f':
                name = nam + '-' + wq + '-monthly-' + '%d'%ll +'.png'

            fig.savefig(pi + var + '/' +  name,dpi=300,bbox_inches='tight')
            plt.close('all')
#            break
#        break
#      break
#    break

