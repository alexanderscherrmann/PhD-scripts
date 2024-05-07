import numpy as np
import os
import pickle
trackpath = '/atmosdyn/michaesp/mincl.era-5/tracks/'

tracks = np.array([])
for d in os.listdir(trackpath):
    if(d.startswith('fi_')):
        if(d[-1]!='s'):
            tracks = np.append(tracks,d)

tracks = np.sort(tracks)
IDs = np.array([])
tracktoid = dict()
for qqq,fdate in enumerate(tracks):
    d = np.loadtxt(trackpath + fdate,skiprows=4)

    tracktoid[fdate] = np.array([])
    ids = np.append(0,np.where((d[1:,-1]-d[:-1,-1])!=0)[0] + 1)
    ids = np.append(ids,len(d[:,-1]))

    for k,i in enumerate(ids[:-1]):
        IDs = np.append(IDs,d[i,-1])
        tracktoid[fdate] = np.append(tracktoid[fdate],d[i,-1])

a, c =np.unique(IDs,return_counts=True)
whichIDdouble = a[np.where(c>1)[0]]

doubleids = dict()
for did in whichIDdouble:
    doubleids[did] = np.array([])
    for date in tracktoid.keys():
        if np.any(tracktoid[date]==did):
            doubleids[did]=np.append(doubleids[did],date)

savedict = dict()
savedict['fullIDarray'] = IDs
savedict['doubleIDarray'] = whichIDdouble
savedict['tracktoiddict'] = tracktoid
savedict['doubleidtotrackdict'] = doubleids

f = open('/home/ascherrmann/double-cyclone-track-ids.txt','wb')
pickle.dump(savedict,f)
f.close()


        
