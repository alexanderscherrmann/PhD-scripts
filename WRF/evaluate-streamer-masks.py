from netCDF4 import Dataset as ds
import numpy as np
import os

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib.colors import BoundaryNorm

minlon = -10
minlat = 30
maxlat = 50
maxlon = 45

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)
lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1
ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/'
for dirs in os.listdir(ps):
  if dirs[-1]!='c':
    ID = dirs +'/'
    if ID!='000542/':
        continue

    d = ds(ps + ID + 'streamer-masks.nc','r')
    mask = d.variables['mask'][:,la0:la1,lo0:lo1]
    
    di = dict()
    var = ['ID','a','lami','lama','lomi','loma','val','maxPV','avPV','avlo','avla','avlow','avlaw','intePV','max10PV','max20PV','maxPVlo','maxPVla']
    maxlen = 0
    for k,f in zip(range(mask.shape[0]),os.listdir(ps + ID)):
        di[k] = dict()
        for va in var:
            di[k][va] = np.array([])
    
        vm = np.unique(mask[k])[1:]
        if len(vm)>maxlen:
            maxlen=len(vm)
        dd = ds(ps + ID + f,'r')
        PV = dd.variables['PV'][0,0,la0:la1,lo0:lo1]
        for v in vm:
            idslat = np.where(mask[k]==v)[0]
            idslon = np.where(mask[k]==v)[1]
            if len(idslat)>=30:
                di[k]['ID'] = np.append(di[k]['ID'],v)
                di[k]['a'] = np.append(di[k]['a'],len(idslat))
                di[k]['lami'] = np.append(di[k]['lami'],np.min(LAT[la0+idslat]))
                di[k]['lama'] = np.append(di[k]['lama'],np.max(LAT[la0+idslat]))
                di[k]['lomi'] = np.append(di[k]['lomi'],np.min(LON[lo0+idslon]))
                di[k]['loma'] = np.append(di[k]['loma'],np.max(LON[lo0+idslon]))
    
                di[k]['avlo'] = np.append(di[k]['avlo'],np.mean(LON[lo0+idslon]))
                di[k]['avla'] = np.append(di[k]['avla'],np.mean(LAT[la0+idslat]))
    
                di[k]['max10PV'] = np.append(di[k]['max10PV'],np.mean(np.sort(PV[idslat,idslon])[-10:]))
                di[k]['max20PV'] = np.append(di[k]['max20PV'],np.mean(np.sort(PV[idslat,idslon])[-20:]))
                di[k]['maxPV'] = np.append(di[k]['maxPV'],np.max(PV[idslat,idslon]))
                di[k]['avPV'] = np.append(di[k]['avPV'],np.mean(PV[idslat,idslon]))
                di[k]['intePV'] = np.append(di[k]['intePV'],np.sum(PV[idslat,idslon]))
    
                di[k]['avlow'] = np.append(di[k]['avlow'],np.sum(LON[lo0+idslon] * PV[idslat,idslon])/di[k]['intePV'][-1])
                di[k]['avlaw'] = np.append(di[k]['avlaw'],np.sum(LAT[la0+idslat] * PV[idslat,idslon])/di[k]['intePV'][-1])           
                
                ar = np.argsort(PV[idslat,idslon])[-10:]
                di[k]['maxPVlo'] = np.append(di[k]['maxPVlo'],np.sum(LON[lo0+idslon[ar]] * PV[idslat[ar],idslon[ar]])/(10 * di[k]['max10PV'][-1]))
                di[k]['maxPVla'] = np.append(di[k]['maxPVla'],np.sum(LAT[la0+idslat[ar]] * PV[idslat[ar],idslon[ar]])/(10 * di[k]['max10PV'][-1]))


    compavlow = []
    compavlaw = []
    compid = []
    uid = np.array([])
    for k in di.keys():
        compavlow.append(di[k]['avlow'])
        compavlaw.append(di[k]['avlaw'])
        compid.append(di[k]['ID'])
   
#    for k,f in zip(range(mask.shape[0]),os.listdir(ps + ID)):
#        if k>0:
#            mask0=mask[k-1]
#            mask1=mask[k]
#            #print(di[k-1]['ID'],di[k]['ID'])
#            for w,vp in enumerate(compid[k-1]):
#                #print('vp',vp)
#                for q,vn in enumerate(compid[k]):
#                    #print('vn',vn)
#                    overlap = np.zeros_like(mask[0])
#                    overlap[mask0==vp]+=1
#                    overlap[mask1==vn]+=1
#                    if np.any(overlap==2):
#                        if len(np.where((overlap==2) & (mask0==vp))[0])/len(np.where((overlap==2) & (mask1==vn))[0])>=0.25 and len(np.where((overlap==2) & (mask0==vp))[0])/len(np.where((overlap==2) & (mask1==vn))[0])<=2:
#                            #print('change %d to %d'%(vn,vp))
#                            compid[k][q] = compid[k-1][w]
#                            mask[k][mask[k]==vn]=vp
        
    for k in list(di.keys())[:-1]:
        for w,la,lo in zip(range(len(compavlaw[k])),compavlaw[k],compavlow[k]):
            for q,la2,lo2 in zip(range(len(compavlaw[k+1])),compavlaw[k+1],compavlow[k+1]):
                r=np.sqrt((la-la2)**2 + (lo-lo2)**2)
                if r <=10:
                    compid[k+1][q] = compid[k][w]

    for k in di.keys():
        uid = np.append(uid,np.unique(di[k]['ID']))
    uid = np.unique(uid)
    
    if ID=='000542/':
        masks = mask
        compids = compid
        dis = di
        uids = uid

    tracks=np.array([])
    for i in uid:
        for k in di.keys():
            if np.any(np.array(compid[k])==i):
                loc = np.where(np.array(compid[k])==i)[0][0]
                if tracks.size==0:
                    tracks = np.array([-168 + k * 3,compavlow[k][loc],compavlaw[k][loc],di[k]['a'][loc],di[k]['avPV'][loc],di[k]['intePV'][loc],di[k]['maxPV'][loc], di[k]['max10PV'][loc],di[k]['max20PV'][loc],di[k]['maxPVlo'][loc],di[k]['maxPVla'][loc],i])
                else:
                    tracks = np.vstack((tracks,np.array([-168 + k * 3,compavlow[k][loc],compavlaw[k][loc],di[k]['a'][loc],di[k]['avPV'][loc],di[k]['intePV'][loc],di[k]['maxPV'][loc], di[k]['max10PV'][loc],di[k]['max20PV'][loc],di[k]['maxPVlo'][loc],di[k]['maxPVla'][loc],i])))
    fmt=['%d','%.2f','%.2f','%d','%.3f','%.3f','%3.f','%.3f','%3.f','%.3f','%3.f','%d']
    np.savetxt(ps + ID + 'tracks-2.txt',tracks,fmt=fmt,delimiter=' ',newline='\n',header='time\tlon\tlat\tarea\tavPV\tintegratedPV\tmaxPV\tmax10PV\tmax20PVi\tmax10PVlon\tmax10PVlat\tID')
    

### plotting purpose
#uv = np.arange(len(uid))+1
#lui = len(uid)
#ci = uv/lui

ID = '000542/'
mask = masks
compid = compids
di = dis
uid = uids
lui = len(uid)
uv = np.arange(len(uid))+1
ci = uv/lui

for k,f in zip(range(mask.shape[0]),os.listdir(ps + ID)):
    
    fig = plt.figure(figsize=(6,4))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
    ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

    dd = ds(ps + ID + f,'r')
    PV = dd.variables['PV'][0,0,la0:la1,lo0:lo1]
    
    for q,ids in enumerate(compid[k]):
        qw = np.where(uid==ids)[0][0]

        ax.contour(LON[lons],LAT[lats],mask[k],levels=[ids],cmap=matplotlib.cm.nipy_spectral,norm=BoundaryNorm(uid,256),linewidths=1)
        ax.scatter(di[k]['avlow'][q],di[k]['avlaw'][q],marker='o',color=matplotlib.cm.nipy_spectral(ci[qw]))
    fig.savefig(ps + ID + 'fig-%s.png'%f[1:],dpi=300,bbox_inches='tight')
    plt.close('all')

