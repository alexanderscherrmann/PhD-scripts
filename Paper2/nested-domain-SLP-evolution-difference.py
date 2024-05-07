# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import matplotlib.gridspec as gridspec

ntracks = '/atmosdyn2/ascherrmann/scripts/WRF/nested-cyclone-tracking/out/'
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ['DJF-clim-max-U-at-300-hPa-0.7-QGPV-new-tracks.txt','DJF-clim-max-U-at-300-hPa-1.4-QGPV-new-tracks.txt','DJF-clim-max-U-at-300-hPa-2.1-QGPV-new-tracks.txt']
new1 = ['DJF-nested-0.7-01-new-tracks.txt','DJF-nested-1.4-01-new-tracks.txt','DJF-nested-2.1-01-new-tracks.txt']
new2 = ['DJF-nested-0.7-02-new-tracks.txt','DJF-nested-1.4-02-new-tracks.txt','DJF-nested-2.1-02-new-tracks.txt']
name = ['DJF-nested-0.7','DJF-nested-1.4','DJF-nested-2.1']
for r,n1,n2,nam in zip(ref,new1,new2,name):

    otrack = np.loadtxt(tracks + r,skiprows=4)
    tra = otrack
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]
    loc = np.where(IDs==2)[0]
    lo = np.where(slp[loc] ==np.min(slp[loc]))[0][0]
    tm = t[loc[lo]]
    dm =helper.simulation_time_to_day_string(tm)
    
    fig = plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0,0])
    ax.plot(1 + t[loc]/24,slp[loc],color='b')
    ax.set_xlabel('simulation time [d]')
    ax.set_xlim(1+t[loc[0]]/24,1+t[loc[-1]]/24)
    ax.set_ylabel('SLP [hPa]')
    ax.set_ylim(990,1015)
    
    tra = np.loadtxt(ntracks + n1,skiprows=4)
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]
    loc = np.where(IDs==2)[0]
    lo = np.where(slp[loc] ==np.min(slp[loc]))[0][0]
    tm = t[loc[lo]]
    dm =helper.simulation_time_to_day_string(tm)
    ax.plot(1 + t[loc]/24,slp[loc],color='purple',linestyle='--')
    
    tra = np.loadtxt(ntracks + n2,skiprows=4)
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]
    loc = np.where(IDs==2)[0]
    ax.plot(1 + t[loc]/24,slp[loc],color='red',linestyle=':')
    ax.legend(['single domain 0.5$^{\circ}$','nested domain 0.5$^{\circ}$','nested domain 0.1$^{\circ}$'],loc='upper right')
    fig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/%s-SLP-evo-difference.png'%nam,dpi=300,bbox_inches="tight")
    plt.close('all')
