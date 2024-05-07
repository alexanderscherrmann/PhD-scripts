import wrf
from netCDF4 import Dataset as ds
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import wrfsims
import helper
import numpy as np
import matplotlib.pyplot as plt


SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/home/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')
LAT = wrf.getvar(ref,'lat')

name = ['AT','MED']
var=['t','slp','PV850','PV300','PV400-200','PV950-700']

#fig,ax = plt.subplots()
#ax.plot([],[],color='dodgerblue')
#ax.plot([],[],color='orange')
#ax.plot([],[],color='r')
#ax.plot([],[],color='peru')
#ax.plot([],[],color='lightcoral')
#ax2=ax.twinx()
for simid, sim in enumerate(SIMS):

    if "-not" in sim:
        continue
    if sim!='DJF-clim' and sim!='DJF-clim-max-U-at-300-hPa-0.3-QGPV' and sim!='DJF-clim-max-U-at-300-hPa-0.5-QGPV':
        continue
    if not os.path.exists(tracks + sim + '-PV-tracks.txt'):
        continue
    
    fig,ax = plt.subplots()
    ax.plot([],[],color='dodgerblue')
    ax.plot([],[],color='orange')
    ax.plot([],[],color='r')
#    ax.plot([],[],color='peru')
#    ax.plot([],[],color='lightcoral')
    ax2=ax.twinx()
    ax3=ax.twiny()

    tra = np.loadtxt(tracks + sim + '-PV-tracks.txt')
    
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]
    pv950_700=tra[:,-2]
    pv400_200=tra[:,-3]
    pv850=tra[:,-4]
    pv300=tra[:,-5]

    locat,locmed = np.where(IDs==1)[0],np.where(IDs==2)[0] 
    loc = [locat,locmed]
    for q,l in enumerate(loc):
        if q==0:
            continue
        tp = t[l]
        tp = tp-tp[np.argmin(slp[l])]
        ax.plot(tp,slp[l],color='dodgerblue')

        ax2.plot(tp,pv300[l],color='orange')
        ax2.plot(tp,pv850[l],color='r')
        #ax2.plot(tp,pv400_200[l],color='peru')
        #ax2.plot(tp,pv950_700[l],color='lightcoral')

    ax.set_xlabel('time to mature stage [h]')
    ax.set_ylabel('SLP [hPa]')
    ax2.set_ylabel('PV [PVU]')
    ax.set_ylim(992,1020)
    ax2.set_ylim(0,8)
    ax3.set_xlabel('simulation time [d]')
    ax3.set_xticks(np.arange(0,227,12))
    ax3.set_xticklabels(np.arange(0,227,12)/24)
    for tl in np.arange(0,227,24):
        ax3.axvline(tl,color='grey',linestyle=':')

    ax3.set_xlim(np.min(t[l]),np.max(t[l]))

    ax.legend(['SLP','PV @ 300 hPa','PV @ 850 hPa'],loc='upper right')# 'av. PV between 400-200 hPa', 'av. PV between 950-700 hPa'],loc='upper right')
    fig.savefig(dwrf + 'image-output/%s-SLP-PV-evol.png'%sim,dpi=300,bbox_inches='tight')
    plt.close(fig)

#ax.set_xlabel('time of mature stage [h]')
#ax.set_ylabel('SLP [hPa]')
#ax2.set_ylabel('PV [PVU]')
#ax.legend(['SLP','PV @ 300 hPa','PV @ 850 hPa', 'av. PV between 400-200 hPa', 'av. PV between 950-700 hPa'],loc='upper left')
#fig.savefig(dwrf + 'image-output/SLP-PV-evol.png',dpi=300,bbox_inches='tight')
#plt.close('all')


