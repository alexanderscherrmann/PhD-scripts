import wrf
from netCDF4 import Dataset as ds
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import wrfsims
import numpy as np
import matplotlib.pyplot as plt


SIMS,ATIDS,MEDIDS = wrfsims.sppt_ids()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

### connect tracks
for simid, sim in enumerate(SIMS):
    print(sim)
    
    medid = np.array(MEDIDS[simid])
    atid = np.array(ATIDS[simid])

    tra = np.loadtxt(tracks + sim + '-filter.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    deepening = tra[:,-2]
    IDs = tra[:,-1]

    if medid.size==0:
        continue

    track = [atid, medid]
    name = ['AT','MED']
    newid = 1
    ftime = np.array([])
    flon= np.array([])
    flat= np.array([])
    fID= np.array([])
    fslp = np.array([])
    foID = np.array([])
    for nm,tr in zip(name,track):

        if tr.size==1:
            loc = np.where(IDs==tr[0])[0]
            newlon = tlon[loc]
            newlat = tlat[loc]
            newslp = slp[loc]
            newtime = t[loc]
            newoID = IDs[loc]

        else: ## combine tracks
            
            newtrack= np.array([])
            newslp = np.array([])
            newlon = np.array([])
            newlat = np.array([])
            combids = np.array([])
            newoID = np.array([])
            idvids = []
            for ids in tr:
                combids = np.append(combids,np.where(IDs==ids)[0])
                idvids.append(np.where(IDs==ids)[0])

            combids = combids.astype(int)
            newtime = np.unique(t[combids])

            for tn in newtime:
                true_array = np.array([])
                tmp_slps = np.array([])
                tmp_lons = np.array([])
                tmp_lats = np.array([])
                tmp_oID = np.array([])
                for ids in idvids:
                    true_array = np.append(true_array,np.any(t[ids]==tn))

                for tru,ids in zip(true_array,idvids):
                    if tru:
                        loc = np.where(t[ids]==tn)[0][0]
                        tmp_slps = np.append(tmp_slps,slp[ids[loc]])
                        tmp_lons = np.append(tmp_lons,tlon[ids[loc]])
                        tmp_lats = np.append(tmp_lats,tlat[ids[loc]])
                        tmp_oID = np.append(tmp_oID,IDs[ids[loc]])

                    else:
                        tmp_slps = np.append(tmp_slps,np.nan)
                        tmp_lons = np.append(tmp_lons,np.nan)
                        tmp_lats = np.append(tmp_lats,np.nan)
                        tmp_oID = np.append(tmp_oID,np.nan)
              
                newslp = np.append(newslp,np.nanmin(tmp_slps))
                newlon = np.append(newlon,tmp_lons[np.nanargmin(tmp_slps)])
                newlat = np.append(newlat,tmp_lats[np.nanargmin(tmp_slps)])
                newoID = np.append(newoID,tmp_oID[np.nanargmin(tmp_slps)])

        ftime = np.append(ftime,newtime)
        flon = np.append(flon,newlon)
        flat = np.append(flat,newlat)
        fslp = np.append(fslp,newslp)
        foID = np.append(foID,newoID)
        newids = np.ones_like(newlon)*newid
        fID = np.append(fID,newids)
        newid+=1

    np.savetxt(tracks + sim + '-new-tracks.txt',np.stack((ftime,flon,flat,fslp,foID,fID),axis=1),fmt='%.2f',newline='\n')

