import wrf
from netCDF4 import Dataset as ds
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import wrfsims
import helper
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
import cmocean
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import re


SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/home/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')
LAT = wrf.getvar(ref,'lat')
lon = LON[0]
lat = LAT[:,0]

name = ['AT','MED']
ids =  [1,2]
seasons = ['DJF','MAM','JJA','SON']
cmap = cmocean.cm.amp
levels = np.arange(10,91,10)
norm = BoundaryNorm(levels,cmap.N)

for sea in seasons:

  for q,na in enumerate(name):
   genesis_density = np.zeros_like(LON)
   counter = 0

   fig=plt.figure(figsize=(8,6))
   gs = gridspec.GridSpec(nrows=1, ncols=1)
   ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
   ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')

   for simid, sim in enumerate(SIMS):
     if sea!=sim[:3]:
         continue
     if "-not" in sim:
         continue
     if not os.path.exists(tracks + sim + '-new-tracks.txt'):
         continue
     
     medid = MEDIDS[simid]
     atid = ATIDS[simid]

     tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
     
     t = tra[:,0]
     tlon,tlat = tra[:,1],tra[:,2]
     slp = tra[:,3]
     IDs = tra[:,-1]
     oID = tra[:,-2]

     if (q==1 and len(medid)==0) or (q==0 and len(atid)==0):
        continue
         
     ### take first 2 track points
     loc = np.where(IDs==ids[q])[0][:1]

     for qq,ts in enumerate(t[loc]):
         dstr = helper.simulation_time_to_day_string(ts)
         flagid = oID[loc[qq]]
         flaglon = tlon[loc[qq]]
         flaglat = tlat[loc[qq]]

         FLAG = ds(tracks + 'flags-' + sim + '/B200012%s.flag01'%dstr)
         idcont = FLAG.variables['IDCONT'][0,:]
         idco = idcont[np.where(lat==flaglat)[0][0],np.where(lon==flaglon)[0][0]]

         genesis_density[idcont==idco]+=1
         counter+=1
         FLAG.close()


   h=ax.contourf(lon,lat,genesis_density/counter*100,cmap=cmap,norm=norm,levels=levels,extend='max')
   cbax = fig.add_axes([0, 0, 0.1, 0.1])
   cbar=plt.colorbar(h, ticks=levels,cax=cbax)
   func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
   fig.canvas.mpl_connect('draw_event', func)

   fig.savefig(dwrf + 'image-output/%s-%s-genesis-map.png'%(sea,na),dpi=300,bbox_inches='tight')
   plt.close('all')


for sea in seasons:

  for q,na in enumerate(name):
   mature_density = np.zeros_like(LON)
   counter = 0

   fig=plt.figure(figsize=(8,6))
   gs = gridspec.GridSpec(nrows=1, ncols=1)
   ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
   ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')

   for simid, sim in enumerate(SIMS):
     if sea!=sim[:3]:
         continue
     if "-not" in sim:
         continue
     if not os.path.exists(tracks + sim + '-new-tracks.txt'):
         continue
     medid = MEDIDS[simid]
     atid = ATIDS[simid]

     tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
     t = tra[:,0]
     tlon,tlat = tra[:,1],tra[:,2]
     slp = tra[:,3]
     IDs = tra[:,-1]
     oID = tra[:,-2]

     if (q==1 and len(medid)==0) or (q==0 and len(atid)==0):
        continue
     loc = np.where((IDs==ids[q]) & (t<192))[0]
     tmpslp = slp[loc]
     ### +- 3 h around mature stage 
#     loc = loc[np.argmin(tmpslp)-1:np.argmin(tmpslp)+1]
     loc = np.array([loc[np.argmin(tmpslp)]])
     for qq,ts in enumerate(t[loc]):
         dstr = helper.simulation_time_to_day_string(ts)
         flagid = oID[loc[qq]]
         flaglon = tlon[loc[qq]]
         flaglat = tlat[loc[qq]]

         if flaglon%0.25!=0 or flaglon%0.5==0:
             flaglon-=flaglon%0.25
             if flaglon%0.5==0:
                 flaglon+=0.25

         if flaglat%0.25!=0 or flaglat%0.5==0:
             flaglat-=flaglat%0.25
             if flaglat%0.5==0:
                 flaglat+=0.25

         FLAG = ds(tracks + 'flags-' + sim + '/B200012%s.flag01'%dstr)
         idcont = FLAG.variables['IDCONT'][0,:]
         idco = idcont[np.where(lat==flaglat)[0][0],np.where(lon==flaglon)[0][0]]

         mature_density[idcont==idco]+=1
         counter+=1
         FLAG.close()


   h=ax.contourf(lon,lat,mature_density/counter*100,cmap=cmap,norm=norm,levels=levels,extend='max')
   cbax = fig.add_axes([0, 0, 0.1, 0.1])
   cbar=plt.colorbar(h, ticks=levels,cax=cbax)
   func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
   fig.canvas.mpl_connect('draw_event', func)

   fig.savefig(dwrf + 'image-output/%s-%s-mature-map.png'%(sea,na),dpi=300,bbox_inches='tight')
   plt.close('all')


ampls = [0.7,1.4,2.1]

for sea in seasons:
 for amps in ampls:
  for q,na in enumerate(name):
   mature_density = np.zeros_like(LON)
   counter = 0
   if sea!='JJA' or amps!=0.7 or na!='AT':
       continue

   fig=plt.figure(figsize=(8,6))
   gs = gridspec.GridSpec(nrows=1, ncols=1)
   ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
   ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')

   for simid, sim in enumerate(SIMS):
     if sea!=sim[:3]:
         continue
     if "-not" in sim:
         continue
     if not os.path.exists(tracks + sim + '-new-tracks.txt'):
         continue


     strings=np.array([])
     for l in re.findall(r"[-+]?(?:\d*\.\d+|\d+)",sim):
        strings = np.append(strings,float(l[1:]))

     if strings.size==0:
         continue

     amp = float(strings[-1])
     if amps!=amp:
         continue

     medid = MEDIDS[simid]
     atid = ATIDS[simid]

     tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
     t = tra[:,0]
     tlon,tlat = tra[:,1],tra[:,2]
     slp = tra[:,3]
     IDs = tra[:,-1]
     oID = tra[:,-2]

     if (q==1 and len(medid)==0) or (q==0 and len(atid)==0):
        continue

     loc = np.where((IDs==ids[q]) & (t<192))[0]
     tmpslp = slp[loc]
     ### +- 3 h around mature stage
#     loc = loc[np.argmin(tmpslp)-1:np.argmin(tmpslp)+1]
     loc = np.array([loc[np.argmin(tmpslp)]])
     for qq,ts in enumerate(t[loc]):
         dstr = helper.simulation_time_to_day_string(ts)
         flagid = oID[loc[qq]]
         flaglon = tlon[loc[qq]]
         flaglat = tlat[loc[qq]]

         if flaglon%0.25!=0 or flaglon%0.5==0:
             flaglon-=flaglon%0.25
             if flaglon%0.5==0:
                 flaglon+=0.25

         if flaglat%0.25!=0 or flaglat%0.5==0:
             flaglat-=flaglat%0.25
             if flaglat%0.5==0:
                 flaglat+=0.25
         print(sim,flagid,flaglon,flaglat) 
         FLAG = ds(tracks + 'flags-' + sim + '/B200012%s.flag01'%dstr)
         idcont = FLAG.variables['IDCONT'][0,:]
         idco = idcont[np.where(lat==flaglat)[0][0],np.where(lon==flaglon)[0][0]]

         mature_density[idcont==idco]+=1
         counter+=1
         FLAG.close()

   h=ax.contourf(lon,lat,mature_density/counter*100,cmap=cmap,norm=norm,levels=levels,extend='max')
   cbax = fig.add_axes([0, 0, 0.1, 0.1])
   cbar=plt.colorbar(h, ticks=levels,cax=cbax)
   func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
   fig.canvas.mpl_connect('draw_event', func)

   fig.savefig(dwrf + 'image-output/%s-%s-%.1f-mature-map.png'%(sea,na,amps),dpi=300,bbox_inches='tight')
   plt.close('all')
