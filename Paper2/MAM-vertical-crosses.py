import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from netCDF4 import Dataset as ds
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
import wrf
import cartopy.crs as ccrs
import cartopy
from dypy.intergrid import Intergrid
from mpl_toolkits.basemap import Basemap
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
import helper
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import wrfsims
import matplotlib.gridspec as gridspec
import os

SIMS,ATIDS,MEDIDS = wrfsims.sppt_ids()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]

cmap,levels,norm,ticklabels=PV_cmap2()

seas = np.array(['DJF','MAM','JJA','SON'])
amps = np.array([0.7,1.4,2.1]).astype(str)
#intensetime=['06_12','06_12','06_12']
intensetime=['08_12','06_12','05_12']
dis = 500

ymin = 100.
ymax = 1000.

deltas = 5.
mvcross    = Basemap()
initl, = mvcross.drawgreatcircle(-1 * helper.convert_radial_distance_to_lon_lat_dis_new(dis,0),0,helper.convert_radial_distance_to_lon_lat_dis_new(dis,0),0,del_s=deltas)
initpa = initl.get_path()
initlop,initlap = mvcross(initpa.vertices[:,0],initpa.vertices[:,1], inverse=True)
initdim = len(initlop)

pappath = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'


sims,atttt,meddd=wrfsims.upper_ano_only()
for sim in sims:
    if 'not' in sim or not 'MAM' in sim or sim[:-4]=='clim':
        continue
    SIMS=np.append(np.array(SIMS),sim)

Pres=np.arange(100,1016,25)
counter=dict()
PVn = dict()
pvchart=dict()
ampcounter=dict()
q=0
for simid,sim in enumerate(SIMS):
    if not 'MAM' in sim:
        continue

    sea = sim[:3]

    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]

    loc = np.where(IDs==2)[0]
    lo = np.where(slp[loc] ==np.min(slp[loc]))[0][0]
    tm = t[loc[lo]]
    dm =helper.simulation_time_to_day_string(tm)

    if not os.path.isfile(dwrf + sim + '/wrfout_d01_2000-12-%s:00:00'%dm):
        lo=lo-1
        tm = t[loc[lo]]
        dm =helper.simulation_time_to_day_string(tm)
    lonc,latc = tlon[loc[lo]],tlat[loc[lo]]

    dm =helper.simulation_time_to_day_string(tm)
    dlon = helper.convert_radial_distance_to_lon_lat_dis_new(dis,latc)

    lon_start = lonc-dlon
    lon_end   = lonc+dlon
    lat_start = latc
    lat_end   = latc

    line,      = mvcross.drawgreatcircle(lon_start, lat_start, lon_end, lat_end, del_s=deltas)
    path       = line.get_path()
    lonp, latp = mvcross(path.vertices[:,0], path.vertices[:,1], inverse=True)
    dimpath    = len(lonp)

    
    dimpath1=dimpath
    data = ds(dwrf + sim + '/wrfout_d01_2000-12-%s:00:00'%dm)
    pv = wrf.getvar(data,'pvo')
    P = wrf.getvar(data,'pressure')
    lon=wrf.getvar(data,'lon')[0]
    lat=wrf.getvar(data,'lat')[:,0]

    for time,amp in zip(intensetime,amps):
        if amp in sim:
            tt=time

    da=ds(dwrf+sim+'/wrfout_d01_2000-12-%s:00:00'%tt)
    PPV = wrf.getvar(da,'pvo')
    PPP = wrf.getvar(da,'pressure')

    pv300=wrf.interplevel(PPV,PPP,300,meta=False)
    
    x,y=np.where(abs(lon-lonc)==np.min(abs(lon-lonc)))[0][0],np.where(abs(lat-latc)==np.min(abs(lat-latc)))[0][0]

    Var = pv
    Var1 = Var
    vcross = np.zeros(shape=(Var.shape[0],dimpath))
    vcross_p= np.zeros(shape=(Var.shape[0],dimpath))
    bottomleft = np.array([LAT[0], LON[0]])
    topright   = np.array([LAT[-1], LON[-1]])


    for k in range(Var.shape[0]):
        f_vcross     = Intergrid(Var[k,:,:], lo=bottomleft, hi=topright, verbose=0)
        f_p3d_vcross   = Intergrid(P[k,:,:], lo=bottomleft, hi=topright, verbose=0)
        for i in range(dimpath):
            vcross[k,i]     = f_vcross.at([latp[i],lonp[i]])
            vcross_p[k,i]   = f_p3d_vcross.at([latp[i],lonp[i]])

    if q==0:
        q=1
        for amp in amps:
            counter[amp]=np.zeros((Pres.size,dimpath))
            PVn[amp]=np.zeros_like(counter[amp])
            PRES = np.ones(PVn[amp].shape)*Pres[:,None]
            ampcounter[amp]=0
            
            pvchart[amp]=np.zeros_like(pv300[y-19:y+20,x-19:x+20])


    maxpres = np.max(vcross_p,axis=0)
    for amp in amps:
        if amp in sim:
#            if ampcounter[amp]<1:
            pvchart[amp]+=pv300[y-19:y+20,x-19:x+20]
            ampcounter[amp]+=1
            for i in range(initdim):
               for l,u in enumerate(Pres):
                   if u>maxpres[i]:
                       continue
                   else:
                       counter[amp][l,i]+=1
                       PVn[amp][l,i]+=vcross[np.where(abs(vcross_p[:,i]-u)==np.min(abs(vcross_p[:,i]-u)))[0][0],i]


    vcross_p1=vcross_p
    vcross1 = vcross


    fig = plt.figure(figsize=(6,4))
    gs = gridspec.GridSpec(nrows=1, ncols=1)

    ax = fig.add_subplot(gs[0,0])
    
    xcoord = np.zeros(shape=(vcross.shape[0],dimpath1))
    for x in range(Var1.shape[0]):
        xcoord[x,:] = np.array([ i*deltas-dis for i in range(dimpath1) ])

    h = ax.contourf(xcoord, vcross_p1, vcross1, levels = levels, cmap = cmap, norm=norm, extend = 'both')

    ax.set_ylabel('Pressure [hPa]', fontsize=12)
    ax.set_ylim(bottom=ymin, top=ymax)

    ax.set_xlim(-500,500)
    ax.set_xticks(ticks=np.arange(-500,501,250))
    ax.set_xticklabels(labels=np.append(np.arange(-500,500,250),'km'))
    ax.invert_yaxis()

    pos = fig.get_axes()[-1].get_position()
    cbax = fig.add_axes([pos.x0+pos.width,pos.y0,0.01,pos.height])
    cbar=plt.colorbar(h, ticks=levels,cax=cbax)
    cbar.ax.set_yticklabels(labels=np.append(ticklabels[:5],np.append(np.array(ticklabels[5:-1]).astype(int),'PVU')))

    fig.savefig(pappath + '/MAM-crosses/vcross-%s-%s.png'%(sim,dm),dpi=300,bbox_inches='tight')
    plt.close(fig)

#for amp in amps:
#    fig = plt.figure(figsize=(6,4))
#
#    ax = fig.add_subplot(gs[0,0])
#
#    xcoord = np.zeros(shape=(PVn[amp].shape[0],dimpath1))
#    for x in range(PRES.shape[0]):
#        xcoord[x,:] = np.array([ i*deltas-dis for i in range(dimpath1) ])
#
#    h = ax.contourf(xcoord, PRES, PVn[amp]/counter[amp], levels = levels, cmap = cmap, norm=norm, extend = 'both')
#    ax.set_ylabel('Pressure [hPa]', fontsize=12)
#    ax.set_ylim(bottom=ymin, top=ymax)
#
#    ax.set_xlim(-500,500)
#    ax.set_xticks(ticks=np.arange(-500,501,250))
#    ax.set_xticklabels(labels=np.append(np.arange(-500,500,250),'km'))
#    ax.invert_yaxis()
#
#    pos = fig.get_axes()[-1].get_position()
#    cbax = fig.add_axes([pos.x0+pos.width,pos.y0,0.01,pos.height])
#    cbar=plt.colorbar(h, ticks=levels,cax=cbax)
#    cbar.ax.set_yticklabels(labels=np.append(ticklabels[:5],np.append(np.array(ticklabels[5:-1]).astype(int),'PVU')))
#
#    fig.savefig(pappath + '/MAM-crosses/composite-vcross-%s.png'%(amp),dpi=300,bbox_inches='tight')


fig = plt.figure(figsize=(12,4))
gs = gridspec.GridSpec(nrows=1, ncols=3)

for q,amp in enumerate(amps):
    ax=fig.add_subplot(gs[0,q])
    print(ampcounter[amp])
    ax.contourf(np.arange(-55*19,55*20,55),np.arange(-55*19,55*20,55),pvchart[amp]/ampcounter[amp], levels = levels, cmap = cmap, norm=norm, extend = 'both')
    ax.scatter(0,0,marker='o',color='k',s=10)


pos = fig.get_axes()[-1].get_position()
cbax = fig.add_axes([pos.x0+pos.width,pos.y0,0.01,pos.height])
cbar=plt.colorbar(h, ticks=levels,cax=cbax)
cbar.ax.set_yticklabels(labels=np.append(ticklabels[:5],np.append(np.array(ticklabels[5:-1]).astype(int),'PVU')))
fig.savefig(pappath + '/MAM-crosses/composite-pvchart-mature-stage.png',dpi=300,bbox_inches='tight')

plt.close('all')
