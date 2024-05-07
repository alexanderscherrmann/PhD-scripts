import pandas as pd
import os
ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/'
dp = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
we = 'sevendaypriormature'
wi = 'intense-cyclones.csv'
seasons = ['DJF','SON']

for lev in [250,350,400,450]:
 for sea in seasons:
    sel = pd.read_csv(dp + sea + '-' + wi)
    for q,d,ID in zip(range(200),sel[we].values,sel['ID'].values):
        print(ID)
        if not os.path.isdir(ps + '%06d'%ID):
            os.mkdir(ps + '%06d'%ID)
        if not os.path.isdir(ps + '%06d'%ID + '/%d'%lev):
            os.mkdir(ps + '%06d'%ID + '/%d'%lev)

        for k in range(73):
            y = int(d[:4])
            m = int(d[4:6])
            dd = int(d[6:8])
            h = int(d[9:])+3
            if (m<8 and m%2==1) or (m>=8 and m%2==0):
                md=31
            else:
                md=30
            if m==2:
                md=28
                if y%4==0:
                    md = 29

            if h>=24:
                h-=24
                dd+=1
                if dd>md:
                    dd-=md
                    m+=1
                    if m==13:
                        m=1
                        y+=1

            d = '%04d%02d%02d_%02d'%(y,m,dd,h)
            if os.path.isfile(ps + '%06d/%d/D%s'%(ID,lev,d)):
                continue
            os.chdir(ps + '%06d/%d/'%(ID,lev))
            os.system("clim-e5 %s PV@%dhPa.dump test.nc"%(d,lev))


