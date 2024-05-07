import numpy as np
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import os
import pickle
import xarray as xr
import matplotlib
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt

fig, ax= plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.coastlines()

p = '/atmosdyn2/ascherrmann/011-all-ERA5/data/'
trackpath = '/atmosdyn/michaesp/mincl.era-5/tracks/'

tracks = np.array([])
for d in os.listdir(trackpath):
    if(d.startswith('fi_')):
        if(d[-1]!='s'):
            tracks = np.append(tracks,d)

tracks = np.sort(tracks)

regions = ['MED','NA','SA','NP','NP','SP','SP','IO']
boundaries = ['lon','lat']

NAlatup = 83
SB = -83

LONR = [np.array([-5,2,42]),#MED
       np.array([-98,-94,-90,-85,-75,-5,15]),#NA
       np.array([-67,20]),#SA
       np.array([-180,-98,-94,-85,-76]),#NP
       np.array([100,180]),#NP
       np.array([-180,-67]),#SP
       np.array([120,140,180]),#SP
       np.array([20,100,120,140])]#IO

LATR = [np.array([28,42,28,48]),
        np.array([17.5,NAlatup,16,NAlatup,14,NAlatup,10,NAlatup,0,NAlatup,50,NAlatup]),
        np.array([SB,0]),
        np.array([0,68,0,16,0,13,0,8]),
        np.array([0,68]),
        np.array([SB,0]),
        np.array([-17,0,SB,0]),
        np.array([SB,25,SB,0,SB,-17])]

c = ['orange','dodgerblue','green','red','red','purple','purple','grey']
for q, l in enumerate(regions[:]):
    londom = LONR[q]
    latdom = LATR[q]
    for b, o in enumerate(londom[:-1]):
        ax.plot([o,londom[b+1]],[latdom[b*2], latdom[b*2]],color=c[q])
        ax.plot([o,o],[latdom[b*2], latdom[b*2 + 1]],color=c[q])
        ax.plot([londom[b+1],londom[b+1]],[latdom[b*2], latdom[b*2 + 1]],color=c[q])
        ax.plot([o,londom[b+1]],[latdom[b*2 + 1],latdom[b*2 + 1]],color=c[q])


ax.set_extent([-180,180,-90,90], ccrs.PlateCarree())
#ax.gridlines(xlocs=np.arange(-180,180.1,5),ylocs=np.arange(-90,90.1,5),color='k',ls='--')
fig.savefig('/atmosdyn2/ascherrmann/011-all-ERA5/regions.png',dpi=300,bbox_inches='tight')
plt.close('all')

NORO = xr.open_dataset('/atmosdyn2/ascherrmann/009-ERA-5/MED/data/NORO')
LSM = NORO['OL'].values[0]

LON = np.arange(-180,180,0.5)
LAT = np.arange(-90,90.1,0.5)


SLP = []
lon = []
lat = []
dates = []
hourstoSLPmin = []
ID = []
REG = []

savings = ['SLP-','lon-','lat-','ID-','hourstoSLPmin-','dates-','REG-']
yeartmp = 1979
add = 'all'
add = 'deep-over-sea'
for qqq,fdate in enumerate(tracks[346:]):
    year = int(fdate[-6:-2])
    if ((yeartmp-year)!=0):
        for sv in savings:
            savefile = open(p + sv + str(yeartmp) + '-' + add + '.txt',"wb")
            pickle.dump(eval(sv[:-1]),savefile)
            savefile.close()

        SLP = []
        lon = []
        lat = []
        dates = []
        hourstoSLPmin = []
        ID = []
        REG = []

    yeartmp=year
    month = int(fdate[-2:])-1
    days = helper.month_days(year)[month]

    d = np.loadtxt(trackpath + fdate,skiprows=4)
    ids = np.append(0,np.where((d[1:,-1]-d[:-1,-1])!=0)[0] + 1)
    ids = np.append(ids,len(d[:,-1]))

    for k,i in enumerate(ids[:-1]):
        indomain = 0
#        if np.min(d[i:ids[k+1],3])>1000:
#            continue
        
        SLPmin = np.where(d[i:ids[k+1],3] == np.min(d[i:ids[k+1],3]))[0][-1]

        lonmin = d[i:ids[k+1],1][SLPmin]
        latmin = d[i:ids[k+1],2][SLPmin]

        if lonmin>179.749:
            lonmin=0
        lonmin = LON[np.where(abs(LON-lonmin)==np.min(abs(LON-lonmin)))[0][0]]
        latmin = LAT[np.where(abs(LAT-latmin)==np.min(abs(LAT-latmin)))[0][0]]

        d[i:ids[k+1],1][SLPmin] = lonmin
        d[i:ids[k+1],2][SLPmin] = latmin
            
        LONID = np.where(LON==lonmin)[0][0]
        LATID = np.where(LAT==latmin)[0][0]
        if LSM[LATID,LONID]!=0:
            continue

        lonz = d[i:ids[k+1],1][0]
        latz = d[i:ids[k+1],2][0]
        lone = d[i:ids[k+1],1][-1]
        late= d[i:ids[k+1],2][-1]

        for q, r in enumerate(regions):
            londom=LONR[q]
            latdom=LATR[q]

            for b,o in enumerate(londom[:-1]):
                if indomain==1:
                    continue
                if ((lonmin>=o) & (lonmin<=londom[b+1]) &
                    (latmin>=latdom[2*b]) & (latmin<=latdom[2*b+1])):
#                    (lonz>=londom[0]) & (lonz<=londom[-1]) & 
#                    (lone>=londom[0]) & (lone<=londom[-1]) &
#                    (latz>=latdom[2*b]) & (latz<=latdom[2*b+1]) &
#                    (late>=latdom[2*b]) & (late<=latdom[2*b+1])):

                    indomain = 1
                    REG.append(r)
                    ID.append(d[i,-1])

        if indomain==1:
         SLP.append(np.array(d[i:ids[k+1],3]))
         lon.append(np.array(d[i:ids[k+1],1]))
         lat.append(np.array(d[i:ids[k+1],2]))
         hourstoSLPmin.append(np.array(d[i:ids[k+1],0]-d[i:ids[k+1],0][SLPmin]))

         tmp = np.array([])
         for u,v in enumerate(d[i:ids[k+1],0]):
             # if on first days of next month
             if((np.floor(v/24).astype(int) + 1)>days):
                 #if next month is next year
                 if ((int(fdate[-2:]) + 1)>12):
                    #                           add 1 to the year    # so next month -12 in the next year
                    tmp = np.append(tmp,str(int(fdate[-6:-2]) + 1) + '%02d'%(int(fdate[-2:])+1-12) +
                            #days=30, v/24 = 30.5 -> 30-30 + 1          # add the hours of next day
                            '%02d'%(np.floor(v/24).astype(int)-days+1) + '_%02d'%(v%24).astype(int))
                 else:
                     #                  same year               # next month
                    tmp = np.append(tmp,fdate[-6:-2] + '%02d'%(int(fdate[-2:])+1) +
                            #  same as above
                         '%02d'%(np.floor(v/24).astype(int)-days + 1) + '_%02d'%(v%24).astype(int))


             else:
                 tmp = np.append(tmp,str(fdate[-6:]) + '%02d'%(np.floor(v/24).astype(int)+1) +
                         '_%02d'%(v%24).astype(int))
         dates.append(tmp)


for sv in savings:
    savefile = open(p + sv + str(yeartmp) + '-' + add + '.txt',"wb")
    pickle.dump(eval(sv[:-1]),savefile)
    savefile.close()
