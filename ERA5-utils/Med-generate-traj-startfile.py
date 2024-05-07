import numpy as np
import xarray as xr
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import pickle
import os

p = '/home/ascherrmann/009-ERA-5/MED/'

Lat = np.arange(-90,90.1,0.5)
Lon = np.arange(-180,180.1,0.5)

savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']

SLP = []
lon = []
lat = []
ID = []
hourstoSLPmin = []
dates = []

var = [SLP, lon, lat, ID, hourstoSLPmin, dates]

for u,x in enumerate(savings):
    f = open(p + 'traj/' +  x + 'furthersel.txt',"rb")
    var[u] = pickle.load(f)
    f.close()

SLP = var[0]
lon = var[1]
lat = var[2]
ID = var[3]
hourstoSLPmin = var[4]
dates = var[5]

maturedates = np.array([])
clat = np.zeros(len(ID))
clon = np.zeros(len(ID))
for l in range(len(ID[:])):
    ids = np.where(hourstoSLPmin[l]==0)[0][-1]
    ##check correct date format
    if len(dates[l][ids])==11:
    ##
        lat[l][ids] = Lat[np.where(abs(Lat-lat[l][ids])==np.min(abs(Lat-lat[l][ids])))[0]]
        lon[l][ids] = Lon[np.where(abs(Lon-lon[l][ids])==np.min(abs(Lon-lon[l][ids])))[0]]

        maturedates = np.append(maturedates,dates[l][ids])
        clat[l] = np.where(Lat==np.round(lat[l][ids],1))[0][0]
        clon[l] = np.where(Lon==np.round(lon[l][ids],1))[0][0]

for k,t in enumerate(maturedates[:]):
    yyyy = int(t[0:4])
    MM = int(t[4:6])
    DD = int(t[6:8])
    hh = int(t[9:])
    if (MM>12):
        yyyy+=1
        MM-=12
        t = str(yyyy) + '%02d'%MM + '%02d'%DD + '_' + '%02d'%hh

#    if (os.path.isfile(p + 'ctraj/trastart-mature-' + t + '-ID-' + '%06d'%(int(ID[k][0])) + '.txt')):
#        continue

#    else:
    if True:
        ana_path = '/home/era5/cdf/%d/%02d/'%(yyyy,MM)
        sfile = ana_path + 'S' + t
        pfile = ana_path +'/P' + t

        s = xr.open_dataset(sfile)
#        clat2 = helper.radial_ids_around_center_calc_ERA5(200)[1] #+ clat[k]
#        clon2 = helper.radial_ids_around_center_calc_ERA5(200)[0] #+ clon[k]

        CLONIDS,CLATIDS = helper.ERA5_radial_ids_correct(200,Lat[clat[k].astype(int)])

        clon2=CLONIDS + clon[k]
        clat2=CLATIDS + clat[k]

#        clon2=clon2+clon[k]
#        clat2=clat2+clat[k]

        clat2 = clat2.astype(int)
        clon2 = clon2.astype(int)

        pt = np.array([])
        plat = np.array([])
        plon = np.array([])
        PS = s.PS.values[0,clat2,clon2]
        pv = s.PV.values[0]
    
        for l in range(len(clat2)):
            P = 0.01 * s.hyam.values[137-98:] + s.hybm.values[137-98:] * PS[l]
            pid = np.where((P>=700) & (P<=975) & (pv[:,clat2[l],clon2[l]]>=0.75))[0]
            for i in pid:
    #            if(pv[l,i]>0.3):
                   pt = np.append(pt,P[i])
                   plat = np.append(plat,Lat[clat2[l]])
                   plon = np.append(plon,Lon[clon2[l]])
    
        save = np.zeros((len(pt),4))
        save[:,1] = plon
        save[:,2] = plat
        save[:,3] = pt
        if (len(pt)>0):
            np.savetxt(p + 'ctraj/trastart-mature-' + t + '-ID-' + '%06d'%(int(ID[k][0])) + '.txt',save,fmt='%f', delimiter=' ', newline='\n')





