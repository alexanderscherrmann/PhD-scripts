import numpy as np
import pickle
import matplotlib.pyplot as plt
from netCDF4 import Dataset as ds
import pandas as pd
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'

data = pd.read_csv('/atmosdyn2/ascherrmann/011-all-ERA5/data/pandas-basic-data-all-deep-over-sea-12h.csv')

minlon = -50
minlat = 30
maxlat = 85
maxlon = 45

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

mth = 5
gth = 40
lth = 60
checkval = 0.8
minn = 300


#cv = [0.8,0.9,0.8,0.8,0.9]
#gt = [40,40,40,50,50]
#mt = [2,2,5,5,5]
cv = [0.8]
gt = [50]
mt = [0]
lth = 57
sizecheck=0.5

MONTHS = data['months'].values
IDs = data['ID'].values

for mth,gth,checkval in zip(mt,gt,cv):
    f = open(ps +'test-mature-size-%.1f-ridge-streamer-types-%.1f-%d-more-than-%d-in-%d-%d.txt'%(sizecheck,checkval,minn,mth,gth,lth),'rb')
    d = pickle.load(f)
    f.close()
    
    f = open(ps + 'test-mature-size-%.1f-streamer-types-%.1f-%d-more-than-%d-in-%d-%d.txt'%(sizecheck,checkval,minn,mth,gth,lth),'rb')
    s = pickle.load(f)
    f.close()
        
    for k in s.keys():
        print(len(s[k]))
        if len(s[k])>=20:
            months = np.array([])
            roverlap = np.zeros((len(lats),len(lons)))
            soverlap = np.zeros((len(lats),len(lons)))
            le = len(s[k])
    
            for l in s[k]:
                if l==325832:
                    S = ds(ps + '325832/300/old-mask.nc','r')
                    R = ds(ps + '325832/300/ridge-mask.nc','r')
    
                    roverlap[R.variables['mask'][56,la0:la1,lo0:lo1]==4992]+=1
                    soverlap[S.variables['mask'][56,la0:la1,lo0:lo1]==3777]+=1
                    continue
                months = np.append(months,MONTHS[np.where(IDs==int(l))[0][0]])
                pos = np.array([])
                lab = np.array([])
                for u in d[l]['300']['streamer']:
                    pos = np.append(pos,u[1])
                    lab = np.append(lab,u[2])
                m = np.argmin(abs(pos-56))
                scm = pos[m].astype(int)
                scl = lab[m].astype(int)
                
                pos = np.array([])
                lab = np.array([])
                for u in d[l]['300']['ridge']:
                    pos = np.append(pos,u[1])
                    lab = np.append(lab,u[2])
                m = np.argmin(abs(pos-56))
                rcm = pos[m].astype(int)
                rcl = lab[m].astype(int)
    
                S = ds(ps + '%06d/300/streamer-mask.nc'%(int(l)),'r')
                R = ds(ps + '%06d/300/ridge-mask.nc'%(int(l)),'r')
    
                smask = S.variables['mask'][scm,la0:la1,lo0:lo1]
                rmask = R.variables['mask'][rcm,la0:la1,lo0:lo1]
    
                roverlap[rmask==rcl]+=1
                soverlap[smask==scl]+=1
    
                S.close()
                R.close()
                
            print(np.max(roverlap)/le,np.max(soverlap)/le,le)
            fig = plt.figure(figsize=(6,4))
            gs = gridspec.GridSpec(ncols=1, nrows=1)
            ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
            ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
            ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
            ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())
    
    
            ax.contour(LON[lons],LAT[lats],soverlap*100/le,levels=np.arange(5,31,5),colors=['saddlebrown','blue','cyan','purple','orange','red'],linewidths=1,linestyle='-')
            ax.contour(LON[lons],LAT[lats],roverlap*100/le,levels=np.arange(5,31,5),colors=['saddlebrown','blue','cyan','purple','orange','red'],linewidths=1,linestyles='--')
            ax.text(0.05,0.925,'%d'%le,fontsize=6,transform=ax.transAxes,fontweight='bold') 
            fig.savefig(pi + 'mature-ridge-streamer-overlap-for-%06d-%.1f-%d-mth-%d-in-%d-%d.png'%(int(k),checkval,minn,mth,gth,lth),dpi=300,bbox_inches='tight')


            fig,ax = plt.subplots()
            mo,co = np.unique(months,return_counts=True)
            
            ax.bar(mo,co)
            fig.savefig(pi + 'mature-ridge-months-for--%06d-%.1f-%d-mth-%d-in-%d-%d.png'%(int(k),checkval,minn,mth,gth,lth),dpi=300,bbox_inches='tight')
            plt.close('all')

