import numpy as np
import xarray as xr
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper

p = '/home/ascherrmann/009-ERA-5/' + 'MED/cases/'

Lat = np.round(np.linspace(-20,90,221),3)
Lon = np.round(np.linspace(-120,30,301),3)
Lon = np.round(np.linspace(-180,180,721),3)
Lat = np.round(np.linspace(-90,90,361),3)

#maturedates = np.loadtxt(p + 'manos-test-dates.txt',dtype=str)
#data = np.loadtxt(p + 'manos-test-data.txt',dtype=float)
maturedates = ['20051214_04','20180928_05']
clat = [np.where(Lat==35.0)[0][0].astype(int), np.where(Lat==34.5)[0][0].astype(int)]
clon = [np.where(Lon==13.5)[0][0].astype(int), np.where(Lon==19.0)[0][0].astype(int)]
ID = [350393, 524945]
#clat = np.zeros(len(maturedates))
#clon = np.zeros(len(maturedates))

#for k in range(len(maturedates)):
#    clat[k] = np.where(Lat==np.round(data[k,1],3))[0][0]
#    clon[k] = np.where(Lon==np.round(data[k,0],3))[0][0]

for k,t in enumerate(maturedates[:]):
    yyyy = int(t[0:4])
    MM = int(t[4:6])
    DD = int(t[6:8])
    hh = int(t[9:])
#    if (MM==10):
#        continue

#    ana_path = '/home/ascherrmann/009-ERA-5/Manos-test/'
    ana_path = '/atmosdyn/era5/cdf/' + t[0:4] + '/' + t[4:6] + '/'
#    sfile = ana_path + 'PVstuf' + t
    sfile = ana_path + 'S' + t
    pfile = '/atmosdyn/era5/cdf/' + str(yyyy) + '/%02d'%MM +'/P' + t

    s = xr.open_dataset(sfile)

    clat2 = helper.radial_ids_around_center_calc_ERA5(200)[1] + clat[k]
    clon2 = helper.radial_ids_around_center_calc_ERA5(200)[0] + clon[k]

    clat2 = clat2.astype(int)
    clon2 = clon2.astype(int)

    pt = np.array([])
    plat = np.array([])
    plon = np.array([])
    PS = s.PS.values[0,clat2,clon2]
    pv = s.PV.values[0,:,clat2,clon2]

    for l in range(len(clat2)):
        P = 0.01 * s.hyam.values[137-98:] + s.hybm.values[137-98:] * PS[l]
        pid, = np.where((P>=700) & (P<=975))
        for i in pid:
#            if(pv[l,i]>0.6):
               pt = np.append(pt,P[i])
               plat = np.append(plat,Lat[clat2[l]])
               plon = np.append(plon,Lon[clon2[l]])

    save = np.zeros((len(pt),4))
    save[:,1] = plon
    save[:,2] = plat
    save[:,3] = pt

#    np.savetxt(p + 'Manos-test/trastart-mature-2full' + t + '-ID-' + str(int(data[k,-2])) + '.txt',save,fmt='%f', delimiter=' ', newline='\n')
    np.savetxt(p + 'trastart-mature-' + t + '-ID-' + str(ID[k]) + '.txt',save,fmt='%f', delimiter=' ', newline='\n')




