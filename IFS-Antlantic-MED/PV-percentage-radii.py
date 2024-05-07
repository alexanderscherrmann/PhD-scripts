import numpy as np
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import matplotlib
import matplotlib.pyplot as plt
import pickle

CT = 'ETA'

pload = '/home/ascherrmann/010-IFS/ctraj/' + CT + '/use/'
psave = '/home/ascherrmann/010-IFS/'

rdis=800
f = open(pload + 'PV-data-' +CT + 'dPSP-100-ZB-800PVedge-0.3-%d-correct-distance.txt'%rdis,'rb')
PVdata = pickle.load(f)
f.close()

f = open('/home/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','rb')
td = pickle.load(f)
f.close()

radii = np.arange(0,2100,200)
globalpercentage = np.array([])

LON=np.linspace(-180,180,901)
LAT=np.linspace(0,90,226)

labs = helper.traced_vars_IFS()
pvsum = np.where(labs=='PVRCONVT')[0][0]

env = 'env'
cyc = 'cyc'
split=[cyc,env]
datadi = dict()
H = 48

localpercentage = dict()
for date in PVdata['rawdata'].keys():
    localpercentage[date] = dict()

datadi = PVdata['rawdata']

newloc = np.array([])
for rdis in radii:
#  deltaLONLAT = helper.convert_radial_distance_to_lon_lat_dis(rdis)
  wql = 0
  meandi = dict()
  dipv = dict() ####splited pv is stored here
  dit = dict() ### 1 and 0s, 1 if traj is close to center at given time, 0 if not
  meandi[env] = dict()
  meandi[cyc] = dict()
  for uyt,date in enumerate(datadi.keys()):
    montmp = PVdata['mons'][uyt]
    idtmp = PVdata['ids'][uyt]

    htzeta = td[montmp][idtmp]['hzeta']
    zeroid = np.where(htzeta==0)[0][0]

    htzeta = htzeta[:zeroid+1]
    if htzeta[0]>-6:
        continue

    clat = td[montmp][idtmp]['clat'][:zeroid+1]
    clon = td[montmp][idtmp]['clon'][:zeroid+1]

    newloc = np.append(newloc,date)

    dipv[date]=dict()    #accumulated pv is saved here
    dipv[date][env]=dict()
    dipv[date][cyc]=dict()

    tmpclon= np.array([])
    tmpclat= np.array([])

    ### follow cyclone backwards to find its center
    dit[date] = dict()
    dit[date][env] = np.zeros(datadi[date]['time'].shape)
    dit[date][cyc] = np.zeros(datadi[date]['time'].shape)

    for k in range(0,H+1):
        if(np.where(htzeta==(-k))[0].size):
            tmpq = np.where(htzeta==(-k))[0][0]
            tmpclon = np.append(tmpclon,np.mean(clon[tmpq]))
            tmpclat = np.append(tmpclat,np.mean(clat[tmpq]))
        else:
            ### use boundary position that no traj should be near it
            tmpclon = np.append(tmpclon,860)
            tmpclat = np.append(tmpclat,0)

    ### check every hours every trajecttory whether it is close to the center ###
    for e, h in enumerate(datadi[date]['time'][0,:]):
        tmplon = tmpclon[e].astype(int)
        tmplat = tmpclat[e].astype(int)

        ### center lon and latitude
        CLON = LON[tmplon]
        CLAT = LAT[tmplat]

        ### 30.10.2020 radial distance instead of square
        ### if traj is in circle of 200km set cyc entry to 0, else env entry 1
        for tr in range(len(datadi[date]['time'])):
#            if ( np.sqrt( (CLON-datadi[date]['lon'][tr,e])**2 + (CLAT-datadi[date]['lat'][tr,e])**2) <=  deltaLONLAT):
            if (helper.convert_dlon_dlat_to_radial_dis_new(CLON-datadi[date]['lon'][tr,e],CLAT-datadi[date]['lat'][tr,e],CLAT)<=rdis):
                dit[date][cyc][tr,e]=1
            else:
                dit[date][env][tr,e]=1


    for key in split:
        for k, el in enumerate(labs[pvsum:]):
            dipv[date][key][el] = np.zeros(datadi[date]['time'].shape)
            dipv[date][key][el][:,:-1] = np.flip(np.cumsum(np.flip(abs(datadi[date][el][:,1:] + datadi[date][el][:,:-1])/2.*dit[date][key][:,1:],axis=1),axis=1),axis=1)


        dipv[date][key]['APVTOT'] =np.zeros(datadi[date]['time'].shape)
        for el in labs[pvsum:]:
            dipv[date][key]['APVTOT'] += dipv[date][key][el]

        idp = np.where(datadi[date]['PV'][:,0]>=0.75)[0]
        if wql==0:
                meandi[key] = dipv[date][key]['APVTOT'][idp]
        else:
                meandi[key] = np.concatenate((meandi[key], dipv[date][key]['APVTOT'][idp]),axis=0)


    localpercentage[date][rdis] = np.mean(abs(dipv[date][cyc]['APVTOT'][idp,0])/(abs(dipv[date][cyc]['APVTOT'][idp,0])+abs(dipv[date][env]['APVTOT'][idp,0])))

    wql=10
  globalpercentage = np.append(globalpercentage,np.mean(abs(meandi[cyc][:,0])/(abs(meandi[cyc][:,0]) + abs(meandi[env][:,0]))))
globalpercentage[0] = 0



#for date in newloc:
#    montmp = PVdata['mons'][uyt]
#    idtmp = PVdata['ids'][uyt]
#
#    htzeta = td[montmp][idtmp]['hzeta']
#    zeroid = np.where(htzeta==0)[0][0]
#
#    htzeta = htzeta[:zeroid+1]
#    if htzeta[0]>-6:
#        continue
#
#    locper = np.array([0])
#   
#    for rdis in radii[1:]:
#        locper = np.append(locper,localpercentage[date][rdis])
#   
#    fig, axes = plt.subplots()
#    axes.plot(radii,locper)
#    axes.set_ylabel('total cyclonic PV change [%]')
#    axes.set_xlabel('radius to be considered cyclonic [km]')
#    axes.set_xticks(ticks=np.arange(0,2100,200))
#    axes.axvline(400,color='grey',ls='-')
#    axes.set_xlim(0,2000)
#    axes.set_ylim(0,1)
#    name = 'PV-perc-radii/cyclonic-abs-PV-percentage-' + date + '.png'
#    fig.savefig(psave + name,dpi=300,bbox_inches="tight")
#    plt.close('all')

fig, axes = plt.subplots()
axes.plot(radii,globalpercentage*100)
axes.set_ylabel('total cyclonic PV change [%]')
axes.set_xlabel('distance to be considered cyclonic [km]')
axes.set_xticks(ticks=np.arange(0,2100,200))
axes.axvline(800,color='grey',ls='-')
axes.set_xlim(0,2000)
axes.set_ylim(0,100)
#axes.text(0.05, 0.95, 'e)', transform=axes.transAxes,fontsize=16, fontweight='bold', va='top')
axes.tick_params(labelright=False,right=True)
name = CT + '-cyclonic-composite-abs-PV-percentage-correct-distance.png'
fig.savefig(psave + name,dpi=300,bbox_inches="tight")
plt.close('all')


