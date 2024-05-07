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
from dypy.intergrid import Intergrid
from mpl_toolkits.basemap import Basemap

import matplotlib.gridspec as gridspec
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
from colormaps import PV_cmap2
cmap,levels,norm,ticklabels=PV_cmap2()

SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)

dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/home/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'
image = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'
ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]

### time of slp measure of atlantic cyclone
hac = 12

minlon = -15
maxlon = 50
minlat = 20
maxlat = 60

Dis = 1000

ymin = 100.
ymax = 1000.
deltas = 5.
mvcross    = Basemap()


line = dict()
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

    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]

    loc = np.where(IDs==2)[0]
    lo = np.where(slp[loc] ==np.min(slp[loc]))[0][0]
    tm = t[loc[lo]]
    lonc,latc = tlon[loc[lo]],tlat[loc[lo]]
    dm =helper.simulation_time_to_day_string(tm)
    dlon = helper.convert_radial_distance_to_lon_lat_dis_new(Dis,latc)
    lon_start = lonc-dlon
    lon_end   = lonc+dlon
    lat_start = latc
    lat_end   = latc

    line[sim],      = mvcross.drawgreatcircle(lon_start, lat_start, lon_end, lat_end, del_s=deltas)


fig1 = plt.figure(figsize=(12,9.75))
fig2 = plt.figure(figsize=(12,9.75))
fig3 = plt.figure(figsize=(12,9.75))

fi = dict()
fi['DJF'] = fig1
fi['MAM'] = fig2
fi['SON'] = fig3

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


    path       = line[sim].get_path()
    lonp, latp = mvcross(path.vertices[:,0], path.vertices[:,1], inverse=True)
    dimpath    = len(lonp)


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

    vcross = np.zeros(shape=(PV.shape[0],dimpath))
    vcross_p= np.zeros(shape=(PV.shape[0],dimpath))
    bottomleft = np.array([LAT[0], LON[0]])
    topright   = np.array([LAT[-1], LON[-1]])


    for k in range(PV.shape[0]):
        f_vcross     = Intergrid(PV[k,:,:], lo=bottomleft, hi=topright, verbose=0)
        f_p3d_vcross   = Intergrid(p[k,:,:], lo=bottomleft, hi=topright, verbose=0)
        for i in range(dimpath):
            vcross[k,i]     = f_vcross.at([latp[i],lonp[i]])
            vcross_p[k,i]   = f_p3d_vcross.at([latp[i],lonp[i]])

    xcoord = np.zeros(shape=(PV.shape[0],dimpath))
    for x in range(PV.shape[0]):
       xcoord[x,:] = np.array([ i*deltas-Dis for i in range(dimpath) ])

    ax=fi[sea].add_subplot(gs[row,col])
    h = ax.contourf(xcoord,vcross_p,vcross,cmap=cmap,norm=norm,extend='both',levels=levels)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlim(-1000,1000)
    ax.invert_yaxis()
    ax.set_xticks(ticks=[])
    ax.set_yticks(ticks=[])
    if col==0:
        ax.set_yticks(ticks=np.arange(200,1001,200))
    if row==5:
        ax.set_xticks(ticks=np.arange(-1000,501,500))


    ### repeat for reference amplitude
    if 'clim-max' in sim:
        ax.text(-250,950,'%.1f-center'%amp)
        ax.text(600,200,'%d'%slp[loc[np.argmin(slp[loc])]])
        row +=1
        ax=fi[sea].add_subplot(gs[row,col])
        
        ax.contourf(xcoord,vcross_p,vcross,cmap=cmap,norm=norm,extend='both',levels=levels)
        ax.set_ylim(bottom=ymin, top=ymax)
        ax.invert_yaxis()
        ax.set_xlim(-1000,1000)
        ax.set_xticks(ticks=[])
        ax.set_yticks(ticks=np.arange(200,1001,200))
        if row==5:
            ax.set_xticks(ticks=np.arange(-1000,501,500))
        ax.text(-250,950,'%.1f-center'%amp)
        ax.text(600,200,'%d'%slp[loc[np.argmin(slp[loc])]])

    else:
        ax.text(-250,950,'%.1f-%d-%s'%(amp,dis,pos))
        ax.text(600,200,'%d'%slp[loc[np.argmin(slp[loc])]])

    
for sea in ['DJF','MAM','SON']:
    fi[sea].subplots_adjust(wspace=0,hspace=0,right=0.9)
    cbax = fi[sea].add_axes([0.9, 0.075, 0.0001, 0.85])
    cbar=plt.colorbar(h, ticks=levels,cax=cbax)
    func=resize_colorbar_vert(cbax, cbax, pad=0.0, size=0.015)
    fi[sea].canvas.mpl_connect('draw_event', func)
    
    fi[sea].savefig(image+ sea + '-vertical-PV-maps.png',dpi=300,bbox_inches='tight')

plt.close('all')


