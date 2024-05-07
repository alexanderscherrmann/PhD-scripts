from netCDF4 import Dataset as ds
import numpy as np
import pandas as pd
import pickle


trackp = '/atmosdyn/michaesp/mincl.era-5/tracks/'
df = pd.read_csv('/atmosdyn2/ascherrmann/011-all-ERA5/data/pandas-basic-data-all-deep-over-sea-12h.csv')
df = df.iloc[np.where(df['reg'].values=='MED')[0]]

dID = df['ID'].values
htminSLP = df['htSLPmin'].values
mdates = df['dates'].values
months = df['months'].values

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/'
era5 = '/atmosdyn2/era5/cdf/'
LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

black = [28,42,40,50]
west = [-4,9,30,42]
cen = [8.5,21,27,47]
east = [20.5,40,27,40]

regions = [black,west,cen,east]
counts = dict()
di = dict()
regio = ['black','west','central','east']
sea = ['DJF','SON']

for s in sea:
 di[s] = dict()
 counts[s] = np.zeros(4)
 for re in regio:
    di[s][re] = np.zeros(100)

for md,ht,ID,mo in zip(mdates,htminSLP,dID,months):
    s = 'gen'

    yyyy = md[:4]
    mm = md[4:6]
    
    if mo==12 or mo==1 or mo==2:
        s = 'DJF'
    
    if mo==10 or mo==11 or mo==9:
        s = 'SON'

    if s!='SON' and s!='DJF':
        continue

    slptrack = np.loadtxt(trackp + 'fi_' + yyyy + mm,skiprows=4)
    if not np.any(slptrack[:,-1]==ID):
        mm = int(mm)-1
        if mm<1:
            mm=12
            yyyy = '%d'%(int(yyyy)-1)

        slptrack = np.loadtxt(trackp + 'fi_' + yyyy + '%02d'%mm,skiprows=4)
    loc = np.where(slptrack[:,-1]==ID)[0][6]
    lon,lat = slptrack[loc,1],slptrack[loc,2]
    
    for q,re,reg in zip(range(4),regio,regions):
        if counts[s][q] >= 100:
            continue
        if lon>reg[0] and lon<reg[1] and lat>reg[2] and lat<reg[3]:
            counts[s][q] +=1
            di[s][re][int(counts[s][q])-1] = ID

f = open(ps + '100-region-season.txt','wb')
pickle.dump(di,f)
f.close()




