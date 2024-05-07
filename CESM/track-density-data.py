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
import pickle

#SIMS,ATIDS,MEDIDS = wrfsims.cesm_ids_wrongSKIN()
SIMS,ATIDS,MEDIDS = wrfsims.cesm_ids()
SIMS = np.array(SIMS)

dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]

pappath = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'
se='DJF'
period=['ERA5','2010','2040','2070','2100']
atslpdi = dict()
medslpdi = dict()
nadensity = dict()
meddensity = dict()
simcounterat = dict()
simcountermed = dict()
simdensat = dict()
simdensmed = dict()

for perio in period:
 atslpdi[perio] = dict()
 nadensity[perio] = dict()
 meddensity[perio] = dict()
 medslpdi[perio] = dict()
 simcounterat[perio] = dict()
 simcountermed[perio] = dict()
 simdensat[perio] = dict()
 simdensmed[perio] = dict()
 for amp in ['0.7','1.4','2.1']:
 
  atslpdi[perio][amp] = dict()
  nadensity[perio][amp] = np.zeros((LAT.size,LON.size))
  meddensity[perio][amp] = np.zeros((LAT.size,LON.size))
  medslpdi[perio][amp] = dict()
  simcounterat[perio][amp] = dict()
  simcountermed[perio][amp] = dict()
  simdensat[perio][amp] = dict()
  simdensmed[perio][amp] = dict()
 
  for t in range(3,217,3):
     atslpdi[perio][amp][t] = []
     medslpdi[perio][amp][t] = []
     
  for simid,sim in enumerate(SIMS):
     if not amp in sim or 'wrong' in sim or not perio in sim:
         continue
     
     simcounterat[perio][amp][sim]=0
     simcountermed[perio][amp][sim]=0
     
     medid = np.array(MEDIDS[simid])
 
     try:
         print(tracks + sim + '-new-tracks.txt')
         tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
         print('read')
     except:
         continue

     t = tra[:,0]
     tlon,tlat = tra[:,1],tra[:,2]
     slp = tra[:,3]
     IDs = tra[:,-1]
     loc = np.where(IDs==2)[0]
 
     loc = np.where(IDs==1)[0]
     tmpdens = np.zeros_like(nadensity[perio][amp])
     for qq,tt in enumerate(t[loc]):
         simcounterat[perio][amp][sim]+=1
         atslpdi[perio][amp][tt].append(slp[loc[qq]])
         if tlon[loc[qq]]%0.25!=0:
             tlon[loc[qq]]-=tlon[loc[qq]]%0.25
         if tlat[loc[qq]]%0.25!=0:
             tlat[loc[qq]]-=tlat[loc[qq]]%0.25
 
         if tlon[loc[qq]]%0.5==0:
             tlon[loc[qq]]=-0.25
         if tlat[loc[qq]]%0.5==0:
             tlat[loc[qq]]-=0.25
 #        print('at',sim,tlon[loc[qq]],tlat[loc[qq]])
         loi,lai = np.where(LON==tlon[loc[qq]])[0][0],np.where(LAT==tlat[loc[qq]])[0][0]
         mask = ds(tracks + 'flags-' + sim + '/B200012%s.flag01'%helper.simulation_time_to_day_string(tt),'r')
         clus = mask.variables['IDCLUST'][0]
         nadensity[perio][amp][clus==clus[lai,loi]]+=1
         tmpdens[lai-5:lai+6,loi-5:loi+6]+=1
         mask.close()
 
     savdens = np.zeros_like(tmpdens)
     savdens[tmpdens!=0]+=1
     simdensat[perio][amp][sim] = savdens
     print(sim)
     for i in range(1,3):
      loc = np.where(IDs==i)[0]
      for ttlo,ttla in zip(tlon[loc],tlat[loc]):
         if ttlo>-5 and ttlo<15 and ttla > 25 and ttla<42:
             print(sim)
 
     tmpdens = np.zeros_like(nadensity[perio][amp])
     savdens = np.zeros_like(tmpdens)
 
     for qq,tt in enumerate(t[loc]):
         simcountermed[perio][amp][sim]+=1
         if tlon[loc[qq]]%0.25!=0:
             tlon[loc[qq]]-=tlon[loc[qq]]%0.25
         if tlat[loc[qq]]%0.25!=0:
             tlat[loc[qq]]-=tlat[loc[qq]]%0.25
         if tlon[loc[qq]]%0.5==0:
             tlon[loc[qq]]=-0.25
         if tlat[loc[qq]]%0.5==0:
             tlat[loc[qq]]-=0.25
         medslpdi[perio][amp][tt].append(slp[loc[qq]])
 #        print('med',sim,tlon[loc[qq]],tlat[loc[qq]])
         loi,lai = np.where(LON==tlon[loc[qq]])[0][0],np.where(LAT==tlat[loc[qq]])[0][0]
         mask = ds(tracks + 'flags-' + sim + '/B200012%s.flag01'%helper.simulation_time_to_day_string(tt),'r')
         clus = mask.variables['IDCLUST'][0]
         meddensity[perio][amp][clus==clus[lai,loi]]+=1
         tmpdens[lai-5:lai+6,loi-5:loi+6]+=1
         mask.close()
     savdens[tmpdens!=0]+=1
     simdensmed[perio][amp][sim]=savdens

save = dict()
save['atslp'] = atslpdi
save['medslp'] = medslpdi
save['atdens'] = nadensity
save['meddens'] = meddensity
save['medcount'] = simcountermed
save['atcount'] = simcounterat
save['attrackdens'] = simdensat
save['medtrackdens'] = simdensmed

f = open(dwrf + 'data/CESM-density-data-slp-%s.txt'%se,'wb')
pickle.dump(save,f)
f.close()


