import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from netCDF4 import Dataset as ds
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
import wrf

from dypy.intergrid import Intergrid
from mpl_toolkits.basemap import Basemap
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
import helper
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import wrfsims

SIMS,ATIDS,MEDIDS,MEDIDS2 = wrfsims.nested_ids()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
spaper = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'

tracks = '/atmosdyn2/ascherrmann/scripts/WRF/nested-cyclone-tracking/out/'
otracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]

cmap,levels,norm,ticklabels=PV_cmap2()

seas = np.array(['DJF','MAM','JJA','SON'])
amps = np.array([0.7,1.4,2.1])

ymin = 200.
ymax = 1000.

pres = np.arange(200,1001,25)

#for amps in [0.7, 1.4, 2.1]:
#    fig,ax =plt.subplots()
#    counter = np.zeros((len(pres),initdim))
#    PVn = np.zeros(counter.shape)
#    PRES = np.ones(PVn.shape)*pres[:,None]
vertical_pv = dict()

for distance in [100, 200]:
    avdi = dict()
    counter = dict()
    amps = [0.7,1.4,2.1]
    for am in amps:
        avdi[str(am)] = np.zeros(pres.size)
        counter[str(am)] = np.zeros(pres.size)
    fig,ax=plt.subplots()
    ax.plot([],[],color='dodgerblue')
    ax.plot([],[],color='orange')
    ax.plot([],[],color='red')
    ax.plot([],[],color='grey')
    for simid,sim in enumerate(SIMS):
        if 'DJF-nested' in sim or sim=='nested-test':
            continue
        if not 'DJF' in sim:
            continue
        vertical_pv[sim] = np.zeros(pres.size)

        tra = np.loadtxt(tracks + sim + '-02-new-tracks.txt')
        t = tra[:,0]
        tlon,tlat = tra[:,1],tra[:,2]
        slp = tra[:,3]
        IDs = tra[:,-1]

        loc = np.where(IDs==2)[0]
        lo = np.where(slp[loc] ==np.min(slp[loc]))[0][0]
        tm = t[loc[lo]]
        lonc,latc = tlon[loc[lo]],tlat[loc[lo]]
        dm =helper.simulation_time_to_day_string(tm)

        data = ds(dwrf + sim + '/wrfout_d02_2000-12-%s:00:00'%dm)
        pv = wrf.getvar(data,'pvo')
        P = wrf.getvar(data,'pressure')
        LON,LAT = np.round(wrf.getvar(data,'lon')[0],2),np.round(wrf.getvar(data,'lat')[:,1],2)
        print(sim,latc,lonc)
        print(np.min(abs(LON-lonc)))
        print(np.min(abs(LAT-latc)))
        loi,lai = np.where(abs(LON-lonc)==np.min(abs(LON-lonc)))[0][0],np.where(abs(LAT-latc)==np.min(abs(LON-lonc)))[0][0]


        dlon = LON-lonc
        dlat = LAT-latc

        dist = helper.convert_dlon_dlat_to_radial_dis_new(dlon,dlat,latc)
        lai,loi = np.where(dist<=distance)

        for q,pp in enumerate(pres):
            pvtmp = wrf.interplevel(pv,P,pp,meta=False)
            tmp = np.array([])
            for lo,la in zip(loi,lai):
                tmp = np.append(tmp,pvtmp[la,lo])
            vertical_pv[sim][q] = np.mean(tmp)
        color='grey'
        for am in amps:
            if am==0.7 and str(am) in sim:
                avdi[str(am)] +=vertical_pv[sim]
                counter[str(am)] += 1
                color='dodgerblue'
            if am==1.4 and str(am) in sim:
                avdi[str(am)] +=vertical_pv[sim]
                counter[str(am)] += 1
                color='orange'
            if am==2.1 and str(am) in sim:
                avdi[str(am)] +=vertical_pv[sim]
                counter[str(am)] += 1
                color='red'


        ax.plot(vertical_pv[sim],pres,color=color)

    for am in amps:
        if am==0.7:
            color='dodgerblue'
        if am==1.4:
            color='orange'
        if am==2.1:
            color='red'

        ax.plot(avdi[str(am)]/counter[str(am)],pres,color=color,linewidth=3.)
    ax.axvline(0,color='k',linestyle=':')
    ax.set_xlabel('PV [PVU]', fontsize=12)
    ax.set_ylabel('Pressure [hPa]', fontsize=12)
    ax.set_ylim(ymax,ymin)
    ax.set_xlim(-2,5)

    ax.legend(['QGPV = 0.7','QGPV = 1.4','QGPV = 2.1','other'])
    figname = 'vertical-PV-lines-%s.png'%str(distance)
    fig.savefig(spaper+figname,dpi=300,bbox_inches='tight')
    plt.close('all')

plt.close('all')

SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)

for distance in [100, 200]:
  for sea in ['DJF','MAM','SON']:
    avdi = dict()
    counter = dict()
    amps = [0.7,1.4,2.1]
    for am in amps:
        avdi[str(am)] = np.zeros(pres.size)
        counter[str(am)] = np.zeros(pres.size)
    fig,ax=plt.subplots()
    ax.plot([],[],color='dodgerblue')
    ax.plot([],[],color='orange')
    ax.plot([],[],color='red')
    ax.plot([],[],color='grey')
    for simid,sim in enumerate(SIMS):
        if 'not' in sim:
            continue
        if sim[-4:]=='clim':
            continue
        if not sim.startswith(sea + '-'):
            continue

        vertical_pv[sim] = np.zeros(pres.size)

        tra = np.loadtxt(otracks + sim + '-new-tracks.txt')
        t = tra[:,0]
        tlon,tlat = tra[:,1],tra[:,2]
        slp = tra[:,3]
        IDs = tra[:,-1]

        loc = np.where(IDs==2)[0]
        lo = np.where(slp[loc] ==np.min(slp[loc]))[0][0]
        tm = t[loc[lo]]
        lonc,latc = tlon[loc[lo]],tlat[loc[lo]]
        dm =helper.simulation_time_to_day_string(tm)

        data = ds(dwrf + sim + '/wrfout_d01_2000-12-%s:00:00'%dm)
        pv = wrf.getvar(data,'pvo')
        P = wrf.getvar(data,'pressure')
        LON,LAT = wrf.getvar(data,'lon')[0],wrf.getvar(data,'lat')[:,1]

        dlon = LON-lonc
        dlat = LAT-latc

        dist = helper.convert_dlon_dlat_to_radial_dis_new(dlon,dlat,latc)
        lai,loi = np.where(dist<=distance)

        for q,pp in enumerate(pres):
            pvtmp = wrf.interplevel(pv,P,pp,meta=False)
            tmp = np.array([])
            for lo,la in zip(loi,lai):
                tmp = np.append(tmp,pvtmp[la,lo])
            vertical_pv[sim][q] = np.mean(tmp)
        color='grey'
        for am in amps:
            if am==0.7 and str(am) in sim:
                avdi[str(am)] +=vertical_pv[sim]
                counter[str(am)] += 1
                color='dodgerblue'
            if am==1.4 and str(am) in sim:
                avdi[str(am)] +=vertical_pv[sim]
                counter[str(am)] += 1
                color='orange'
            if am==2.1 and str(am) in sim:
                avdi[str(am)] +=vertical_pv[sim]
                counter[str(am)] += 1
                color='red'


        ax.plot(vertical_pv[sim],pres,color=color)

    for am in amps:
        if am==0.7:
            color='dodgerblue'
        if am==1.4:
            color='orange'
        if am==2.1:
            color='red'

        ax.plot(avdi[str(am)]/counter[str(am)],pres,color=color,linewidth=5.)
    ax.axvline(0,color='k',linestyle=':')
    ax.set_xlabel('PV [PVU]', fontsize=12)
    ax.set_ylabel('Pressure [hPa]', fontsize=12)
    ax.set_ylim(ymax,ymin)
    ax.set_xlim(-2,5)

    ax.legend(['QGPV = 0.7','QGPV = 1.4','QGPV = 2.1','other'])
    figname = '%s-vertical-PV-lines-%s.png'%(sea,str(distance))
    fig.savefig(spaper+figname,dpi=300,bbox_inches='tight')
    plt.close('all')

plt.close('all')
