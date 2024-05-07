import wrf
from netCDF4 import Dataset as ds
import os
import numpy as np
import matplotlib.pyplot as plt

dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/home/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]

lon1,lat1,lon2,lat2 = -8,20,1.5,40
lon3,lat3,lon4,lat4 = 1.5,20,50,48

### time of slp measure of atlantic cyclone
hac = 12

PVpos = [[52,28],[145,84],[93,55],[55,46],[84,59],[73,59]]
names = np.array([])
MINslp = np.array([])
MAXpv = np.array([])

fig1,ax = plt.subplots(figsize=(8,6))
fig2,ax2 = plt.subplots(figsize=(8,6))
fig3,ax3 = plt.subplots(figsize=(8,6))


#legend=['normal','dry','saturated','max wind', 'left exit', 'right entrance']
#colors=['k','saddlebrown','b','k','k','k']
#markers=['o','o','o','o','x','+']
legend=['DJF','MAM','JJA','SON','DJF+low','DJF+Llow']
colors=['b','seagreen','r','saddlebrown','dodgerblue','dodgerblue']
markers=['o','o','o','o','+','x']
ls=' '
for co,ma in zip(colors,markers):
    ax.plot([],[],ls=ls,marker=ma,color=co)
    ax2.plot([],[],ls=ls,marker=ma,color=co)
    ax3.plot([],[],ls=ls,marker=ma,color=co)


counter=1
for d in os.listdir(dwrf):
#    if not d.startswith('DJF-clim-') and not d.startswith('dry-DJF-clim') and not d.startswith('sat-DJF-clim'):
#        continue
    if d=='DJF-clim-max-U-at-300hPa-hourly-2.1QG' or d=='sat-DJF-clim-max-U-at-300hPa-hourly-2.1QG':
        continue
    if not d.startswith('DJF-clim-max') and not d.startswith('MAM-clim-max') and not d.startswith('JJA-clim-max') and not d.startswith('SON-clim-max'):# and not d.startswith('DJF-L300') and not d.startswith('DJF-L500') and not d.startswith('DJF-clim-double'):
        continue

    ic = ds(dwrf + d + '/wrfout_d01_2000-12-01_00:00:00')
    tra = np.loadtxt(tracks + d + '-filter.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]
    minslp=np.array([])
    isd = np.array([])
    locs = []
    slpmin=0
    for ids in np.unique(IDs):
        loc = np.where(IDs==ids)[0]
        if t[loc[0]]<60:
            continue
        if not ( ((tlon[loc[0]]>=lon1) and (tlon[loc[0]]<=lon2) and (tlat[loc[0]]>=lat1) and (tlat[loc[0]]<=lat2) ) or ((tlon[loc[0]]>=lon3) and (tlon[loc[0]]<=lon4) and (tlat[loc[0]]>=lat3) and (tlat[loc[0]]<=lat4)) ):
            continue
        if loc.size<16:
            continue
        dlonlat = np.sqrt((tlon[loc[0]]-tlon[loc[-1]])**2 + (tlat[loc[0]]-tlat[loc[-1]])**2)
        if dlonlat<10:
            continue
        minslp = np.append(minslp,np.min(slp[loc]))
        isd = np.append(isd,ids)
        locs.append(loc)

    if minslp.size==0:
        continue

    ids = isd[np.argmin(minslp)]
    locs = locs[np.argmin(minslp)]
    slpmin = np.min(minslp)
#    locs = np.where( (((tlon>=lon1) & (tlon<=lon2) & (tlat>=lat1) & (tlat<=lat2)) | ((tlon>=lon3) & (tlon<=lon4) & (tlat>=lat3) & (tlat<=lat4))) & (t<=168))[0]

    PV = wrf.getvar(ic,'pvo')
    p = wrf.getvar(ic,'pressure')

    pv = wrf.interplevel(PV,p,300,meta=False)
    maxpv = np.zeros(len(PVpos))
    for q,l in enumerate(PVpos):
        maxpv[q] = pv[l[1],l[0]]

    maxpv = np.max(maxpv)
    mark='o'
    col = 'k'
    if d.startswith('DJF-clim'):
        col = 'b'
    if d.startswith('MAM-clim'):
        col = 'seagreen'
    if d.startswith('JJA-clim'):
        col = 'r'
    if d.startswith('SON-clim'):
        col = 'saddlebrown'
    if d.startswith('DJF-clim-double'):
        mark='+'
        col='dodgerblue'
    if d.startswith('DJF-L300'):
        mark='x'
        col='dodgerblue'
    if d.startswith('DJF-L500'):
        mark='d'
        col='dodgerblue'

    ax.scatter(maxpv,slpmin,color=col,marker=mark)
    ax.text(maxpv,slpmin,'%s-%d'%(d[-5:-2],counter))
    ###
    ###
    ###
    #print(tlon[locs[0]])
    ax3.scatter(tlon[locs[0]],maxpv,color=col,marker=mark)
    ax3.text(tlon[locs[0]],maxpv,'%s-%d'%(d[-5:-2],counter))
    ###
    ### Atlantic cyclone filtering
    ###

    aids = np.array([])
    aslp = np.array([])
    aloc = []

    for ids in np.unique(IDs):
        loc = np.where(IDs==ids)[0]
        if loc.size<30:
            continue
        if t[loc[0]]>27:
            continue
        dlonlat = np.sqrt((tlon[loc[0]]-tlon[loc[-1]])**2 + (tlat[loc[0]]-tlat[loc[-1]])**2)
        if dlonlat<20:
            continue

        aids = np.append(aids,ids)
        t12 = np.where(t[loc]==hac)[0]
        aslp = np.append(aslp,slp[loc[t12]])
#        aslp = np.append(aslp,np.min(slp[loc]))
#        aloc.append(loc)
     
    if aids.size==0 or aslp.size==0:
        continue
    aminslp = np.min(aslp)

    ax2.scatter(aminslp,slpmin,color=col,marker=mark)
    ax2.text(aminslp,slpmin,'%s-%d'%(d[-5:-2],counter))

    print(counter,slpmin,d,aids)
    counter+=1
    
ax.set_xlabel('PV anomaly [PVU]')
ax.set_ylabel('MED cyclone min SLP [hPa]')
ax.legend(legend,loc='upper right')
name = dwrf + 'image-output/MED-cyclone-PV-anomaly-SLP-scatter.png'
fig1.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig1)


ax2.set_xlabel('Atlantic cyclone min SLP [hPa]')
ax2.set_ylabel('MED cyclone min SLP [hPa]')
ax2.legend(legend,loc='upper right')
name = dwrf + 'image-output/Atlantic-MED-cyclone-SLP-scatter-%02d.png'%hac
fig2.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig2)


ax3.set_xlabel('MED cyclone genesis lon [$^{\circ}$]')
ax3.set_ylabel('PV anomaly [PVU]')
ax3.legend(legend,loc='upper right')
name = dwrf + 'image-output/MED-cyclone-lon-start-scatter.png'
fig3.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig2)

