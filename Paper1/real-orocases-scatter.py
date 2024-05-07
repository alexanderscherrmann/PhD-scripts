import numpy as np
import pickle
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cartopy
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs


f = open('/atmosdyn2/ascherrmann/009-ERA-5/MED/ctraj/use/PV-data-dPSP-100-ZB-800-2-400-correct-distance-noro.txt','rb')
data = pickle.load(f)
f.close()

datadi = data['rawdata']
dipv = data['dipv']
noro = data['noro']
oro = data['oro']

df = pd.read_csv('/atmosdyn2/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')
#thresh='1.5PVU'
thresh='075'
df = df.loc[df['ntraj%s'%thresh]>=200]
ID = df['ID'].values

NORO = xr.open_dataset('/atmosdyn2/ascherrmann/009-ERA-5/MED/data/NORO')
LON  = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)
OL = NORO['OL'][0]

clim = np.loadtxt('/atmosdyn2/ascherrmann/009-ERA-5/MED/clim-avPV.txt')
df2 = pd.DataFrame(columns=['avPV'],index=['Year','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
df2['avPV'] = np.append(np.mean(clim),clim)
        
df2 = pd.DataFrame(columns=['avPV'],index=['Year','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
df2['avPV'] = np.append(np.mean(clim),clim)
poroid = np.array([])
oro = data['oro']
djfc = 0
mamc = 0 
jjac = 0 
sonc = 0 
for k in dipv.keys():

    if np.all(ID!=int(k)):
        continue
    i = np.where(ID==int(k))[0][0]
    mon = df['mon'].values[i]


    idp = np.where((datadi[k]['PV'][:,0]>=0.75)&(datadi[k]['P'][:,0]>=700))[0]

    if np.mean(oro[k]['env'][idp,0])<0.15:
        continue

    if np.mean(dipv[k]['env'][idp,0])/np.mean(datadi[k]['PV'][idp,0]-df2['avPV'][mon])<0.25:
        continue

    if np.mean(dipv[k]['env'][idp,0]/(datadi[k]['PV'][idp,0]-df2['avPV'][mon]))>=0.25:

        if np.mean(dipv[k]['env'][idp,0]/(datadi[k]['PV'][idp,0]-df2['avPV'][mon])) * np.mean(oro[k]['env'][idp,0])/np.mean(dipv[k]['env'][idp,0]) >= 0.25:
        
            poroid = np.append(poroid,k)
            if mon=='DEC' or mon=='JAN' or mon=='FEB':
                djfc+=1
            if mon=='MAR' or mon=='APR' or mon=='MAY':
                mamc+=1
            if mon=='JUN' or mon=='JUL' or mon=='AUG':
                jjac+=1
            if mon=='SEP' or mon=='OCT' or mon=='NOV':
                sonc+=1

        
        
print(djfc,mamc,jjac,sonc)
print(poroid.size)
fig, ax = plt.subplots(1,1, subplot_kw=dict(projection=ccrs.PlateCarree()),sharex=True,sharey=False)
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)

counter = np.zeros((361,721))
ax.contour(LON[:-1],LAT,NORO['ZB'][0],levels=np.arange(800,3201,400),linewidths=0.5,colors='purple')
for pi in poroid:
    if pi=='108215' or pi=='119896':
        print('still in')

    pl = np.where(int(pi)==ID)[0][0]
    lo = df['lon'].values[pl]
    la = df['lat'].values[pl]
    Lo = np.where(LON==lo)[0][0]
    La = np.where(LAT==la)[0][0]
    counter[La,Lo]+=1
    ax.scatter(lo,la,color='k',s=2)
#    ax.text(lo,la,'%s'%pi,fontsize=4)

counter2 = np.zeros((361,721))
steps = [[0,0],[0,1],[1,0],[1,1]]
for l in range(0,361):
    for a in range(0,721):
        tmp=[]
        for z,s in steps:
            tmp.append(np.sum(counter[l-1+z:l+z+2,a+s-1:a+s+2]))
        counter2[l,a] = np.mean(tmp)/4

levels = np.array([1,5,10,15,20,25,30])
cf = ax.contour(LON,LAT,counter2,linewidths=[0.5,1.5,0.5,0.5,1.5,0.5,0.5],colors='blue',levels=levels)

ax.set_extent([-10,50,25,50])
fig.savefig('/atmosdyn2/ascherrmann/paper/cyc-env-PV/review/new-noro-oro-cyclones-scatt-noro.png',dpi=300,bbox_inches='tight')
#plt.show()
plt.close(fig)

