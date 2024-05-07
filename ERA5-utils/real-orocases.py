import numpy as np
import pickle
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs


f = open('/home/ascherrmann/009-ERA-5/MED/ctraj/use/PV-data-dPSP-100-ZB-800-2-400-correct-distance.txt','rb')
data = pickle.load(f)
f.close()

datadi = data['rawdata']
dipv = data['dipv']

df = pd.read_csv('/home/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')
df = df.loc[df['ntraj075']>=200]
ID = df['ID'].values

NORO = xr.open_dataset('/home/ascherrmann/009-ERA-5/MED/data/NORO')
LON  = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)
OL = NORO['OL'][0]

clim = np.loadtxt('/home/ascherrmann/009-ERA-5/MED/clim-avPV.txt')
df2 = pd.DataFrame(columns=['avPV'],index=['Year','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
df2['avPV'] = np.append(np.mean(clim),clim)
        
df2 = pd.DataFrame(columns=['avPV'],index=['Year','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
df2['avPV'] = np.append(np.mean(clim),clim)
poroid = np.array([])
oro = data['oro']
for k in dipv.keys():
    if np.all(ID!=int(k)):
        continue
    i = np.where(ID==int(k))[0][0]
    mon = df['mon'].values[i]
    if np.mean(dipv[k]['env'][:,0]/(datadi[k]['PV'][:,0]-df2['avPV'][mon]))>0.5 and np.mean(oro[k]['env'][:,0])>= 0.5 * np.mean(dipv[k]['env'][:,0]) :
        poroid = np.append(poroid,k)
        
        
sid = np.array([])
counter = np.zeros((361,721))
counter2 = np.zeros((361,721))

for pi in poroid:
    pl = np.where(int(pi)==ID)[0][0]
    lo = df['lon'].values[pl]
    la = df['lat'].values[pl]
    Lo = np.where(LON==lo)[0][0]
    La = np.where(LAT==la)[0][0]
    counter[La,Lo]+=1
    if OL[La,Lo]==0:
        sid = np.append(sid,pi)

steps = [[0,0],[0,1],[1,0],[1,1]]
for l in range(0,361):
    for a in range(0,721):
        tmp=[]
        for z,s in steps:
            tmp.append(np.sum(counter[l-1+z:l+z+2,a+s-1:a+s+2]))
        counter2[l,a] = np.mean(tmp)

col = df.columns
dfo = pd.DataFrame(columns=col)
for ids in sid:
    dfo = dfo.append(df.loc[df['ID']==int(ids)])


fig, ax = plt.subplots(1,1, subplot_kw=dict(projection=ccrs.PlateCarree()),sharex=True,sharey=False)
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)

ax.contour(LON[:-1],LAT,NORO['ZB'][0],levels=np.arange(800,3201,400),linewidths=0.5,colors='purple')
ax.contour(LON,LAT,counter2,levels=np.array([2,5,8,14,20,26,32]),linewidths=1,colors='k')
ax.set_extent([-10,50,25,50])
fig.savefig('/home/ascherrmann/009-ERA-5/MED/oro-cyclones.png',dpi=300,bbox_inches='tight')
plt.close(fig)

dfo.to_csv('/home/ascherrmann/009-ERA-5/MED/orocyclones.csv',index=False)

