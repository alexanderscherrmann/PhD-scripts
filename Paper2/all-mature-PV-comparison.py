import wrf
from netCDF4 import Dataset as ds
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import wrfsims
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
from colormaps import PV_cmap2
cmap,levels,norm,ticklabels=PV_cmap2()

SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)

f = open('/atmosdyn2/ascherrmann/013-WRF-sim/precipitation-dict.txt','rb')
prec = pickle.load(f)
f.close()

dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/home/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'
image = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'
ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]

### time of slp measure of atlantic cyclone
hac = 12

names = np.array([])
MINslp = np.array([])
MAXpv = np.array([])

fig1 = plt.figure(figsize=(12,9.75))
fig2 = plt.figure(figsize=(12,9.75))
fig3 = plt.figure(figsize=(12,9.75))

fi = dict()
fi['DJF'] = fig1
fi['MAM'] = fig2
fi['SON'] = fig3


minlon = -15
maxlon = 50
minlat = 20
maxlat = 60

gs = gridspec.GridSpec(nrows=6, ncols=5)

legend=np.array(['DJF','MAM','JJA','SON'])
colors=['b','darkgreen','r','saddlebrown','dodgerblue','dodgerblue','royalblue']
col200km = ['deepskyblue','limegreen','lightcoral','peru']
ENSW = np.array(['west-','north','east-','south'])
ensw = np.array(['W','N','E','S'])

enswmarker = ['1','d','s','x']

markers=['o','o','o','o','+','x','d']
ls=' '

amps = [0.7, 1.4, 2.1]
dist = np.array([200, 400])
var = ['slp','t-genesis','pvampl','precip']

for simid, sim in enumerate(SIMS):
    
    if "-not" in sim:
        continue
    if sim[-4:]=='clim':
        continue
    
    strings=np.array([])
    for l in re.findall(r"[-+]?(?:\d*\.\d+|\d+)",sim):
        strings = np.append(strings,float(l[1:]))

    amp = float(strings[-1])
    if amp!=0.7 and amp!=1.4 and amp!=2.1:
        continue


    if amp==0.7:
        rowadd=0
    if amp==1.4:
        rowadd=2
    if amp==2.1:
        rowadd=4

    sea = sim[:3]
    if sea=='JJA':
        continue
    row = 0
    if np.any(dist==int(strings[0])):
        dis = int(strings[0])
        if dis==200:
            row=0
        else:
            row=1

    print(sim)
    col = 0
    if np.any(ENSW==sim[11:16]):
        col=1+np.where(ENSW==sim[11:16])[0][0]
        pos=ensw[np.where(ENSW==sim[11:16])[0][0]]
    
    row = row + rowadd

    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]

    loc = np.where(IDs==2)[0]
    tmat = t[loc[np.argmin(slp[loc])]]
    tsim = helper.simulation_time_to_day_string(tmat)

    ic=ds(dwrf + sim + '/wrfout_d01_2000-12-%s:00:00'%tsim)
    PV = wrf.getvar(ic,'pvo')
    p = wrf.getvar(ic,'pressure')

    PV300 = wrf.interplevel(PV,p,300,meta=False)

    ax=fi[sea].add_subplot(gs[row,col],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=2, edgecolor='black')
    h = ax.contourf(LON,LAT,PV300,cmap=cmap,norm=norm,extend='both',levels=levels)
    ax.set_extent([minlon,maxlon,minlat,maxlat])
    ax.scatter(tlon[loc[np.argmin(slp[loc])]],tlat[loc[np.argmin(slp[loc])]],color='purple',marker='o') 
    ### repeat for reference amplitude
    if 'clim-max' in sim:
        ax.text(0,21,'%.1f-center'%amp)
        ax.text(35,55,'%d'%slp[loc[np.argmin(slp[loc])]])
        row +=1
        ax=fi[sea].add_subplot(gs[row,col],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=2, edgecolor='black')

        ax.contourf(LON,LAT,PV300,cmap=cmap,norm=norm,extend='both',levels=levels)
        ax.scatter(tlon[loc[np.argmin(slp[loc])]],tlat[loc[np.argmin(slp[loc])]],color='purple',marker='o')
        ax.set_extent([minlon,maxlon,minlat,maxlat])
        ax.text(0,21,'%.1f-center'%amp)
        ax.text(35,55,'%d'%slp[loc[np.argmin(slp[loc])]])
    else:
        ax.text(0,21,'%.1f-%d-%s'%(amp,dis,pos))
        ax.text(35,55,'%d'%slp[loc[np.argmin(slp[loc])]])

for sea in ['DJF','MAM','SON']:
    fi[sea].subplots_adjust(wspace=0,hspace=0,right=0.9)
    cbax = fi[sea].add_axes([0.9, 0.075, 0.0001, 0.85])
    cbar=plt.colorbar(h, ticks=levels,cax=cbax)
    func=resize_colorbar_vert(cbax, cbax, pad=0.0, size=0.015)
    fi[sea].canvas.mpl_connect('draw_event', func)
    
    fi[sea].savefig(image+ sea + '-streamer-mature-maps.png',dpi=300,bbox_inches='tight')

plt.close('all')
##    if sim[4:8]=='clim':
#    axes2[whichax].scatter(aminslp,slpmin,color=col,marker=mark,s=amp*10-3)
#    axes[whichax].scatter(maxpv,slpmin,color=col,marker=mark,s=amp*10-3)
##    axes3[whichax].scatter(atmt,medmt,color=col,marker=mark)
##        ax5.text(atmt,medmt,'%.1f'%(amp))
#
##ax.set_xlabel('PV anomaly [PVU]')
##ax.set_ylabel('MED cyclone min SLP [hPa]')
#axes[0].legend(legend,loc='upper right')
#axes[0].set_ylabel('MED cyclone min SLP [hPa]')
#axes[2].set_ylabel('MED cyclone min SLP [hPa]')
#axes[2].set_xlabel('PV anomaly [PVU]')
#axes[3].set_xlabel('PV anomaly [PVU]')
#fig1.subplots_adjust(wspace=0,hspace=0)
#name = image + 'errorbar2-MED-cyclone-PV-anomaly-SLP-scatter-min-of-all-tracks-in-MED.png'
#fig1.savefig(name,dpi=300,bbox_inches='tight')
#plt.close(fig1)
#
##ax2.set_xlabel('Atlantic cyclone min SLP [hPa]')
##ax2.set_ylabel('MED cyclone min SLP [hPa]')
#axes2[0].legend(legend,loc='upper left')
#axes2[0].set_ylabel('MED cyclone min SLP [hPa]')
#axes2[2].set_ylabel('MED cyclone min SLP [hPa]')
#axes2[2].set_xlabel('Atlantic cyclone min SLP [hPa]')
#axes2[3].set_xlabel('Atlantic cyclone min SLP [hPa]')
#fig2.subplots_adjust(wspace=0,hspace=0)
#name = image + 'errorbar2-Atlantic-MED-cyclone-SLP-scatter-min-of-all-tracks-in-MED.png'
#fig2.savefig(name,dpi=300,bbox_inches='tight')
#plt.close(fig2)
#
##ax3.set_xlabel('PV anomaly [PVU]')
##ax3.set_ylabel('precipitation [m]')
##axes3[0].legend(legend,loc='upper left')
##axes3[0].set_ylabel('PV anomaly [PVU]')
##axes3[2].set_ylabel('PV anomaly [PVU]')
##axes3[2].set_xlabel('precipitation [m]')
##axes3[3].set_xlabel('precipitation [m]')
##fig3.subplots_adjust(wspace=0,hspace=0)
##name = image + 'errorbar2-precipitation-vs-PV-anomaly.png'
##fig3.savefig(name,dpi=300,bbox_inches='tight')
##plt.close(fig3)


