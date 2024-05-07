import numpy as np
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import os
import pickle

p = '/home/ascherrmann/009-ERA-5/MED/'
trackpath = '/atmosdyn/michaesp/mincl.era-5/tracks/'

tracks = np.array([])
for d in os.listdir(trackpath):
    if(d.startswith('fi_')):
        if(d[-1]!='s'):
            tracks = np.append(tracks,d)

tracks = np.sort(tracks)

londom = np.array([-5,2,42])
latbon = np.array([42,48])

LON = np.arange(-180,180,0.5)
LAT = np.arange(-90,90.1,0.5)

SLP = []
lon = []
lat = []
dates = []
hourstoSLPmin = []
ID = []
savings = ['SLP-','lon-','lat-','ID-','hourstoSLPmin-','dates-']
yeartmp = 1979

for t,fdate in enumerate(tracks):
    year = int(fdate[-6:-2])
    if ((yeartmp-year)!=0):
        for sv in savings:
            savefile = open(p + sv + str(yeartmp) + '.txt',"wb")
            pickle.dump(eval(sv[:-1]),savefile)
            savefile.close()
            
        SLP = []
        lon = []
        lat = []
        dates = []
        hourstoSLPmin = []
        ID = []

    yeartmp=year
    month = int(fdate[-2:])-1
    days = helper.month_days(year)[month]

    d = np.loadtxt(trackpath + fdate,skiprows=4)


    ids = np.append(0,np.where((d[1:,-1]-d[:-1,-1])!=0)[0] + 1)
    ids = np.append(ids,len(d[:,-1]))

    for k,i in enumerate(ids[:-1]):
        indomain = 0
        SLPmin = np.where(d[i:ids[k+1],3] == np.min(d[i:ids[k+1],3]))[0][-1]

        lonmin = d[i:ids[k+1],1][SLPmin]
        lonmin = LON[np.where(abs(LON-lonmin)==np.min(abs(LON-lonmin)))[0][0]]
        lonz = d[i:ids[k+1],1][0]

        if ((lonmin>=londom[0]) & (lonmin<=londom[-1])):
            latmin = d[i:ids[k+1],2][SLPmin]
            latmin = LAT[np.where(abs(LAT-latmin)==np.min(abs(LAT-latmin)))[0][0]]
            for b,o in enumerate(londom[:-1]):
                if ((lonmin>=o) & (lonmin<=londom[b+1]) & (latmin>=30) & (latmin<=latbon[b])):
                    indomain=1
    
        if indomain==1:
         SLP.append(np.array(d[i:ids[k+1],3]))
         lon.append(np.array(d[i:ids[k+1],1]))
         lat.append(np.array(d[i:ids[k+1],2]))
         ID.append(np.array(d[i:ids[k+1],-1]))
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
    savefile = open(p + sv + str(yeartmp) + '.txt',"wb")
    pickle.dump(eval(sv[:-1]),savefile)
    savefile.close()
