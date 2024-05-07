import pickle
import numpy as np
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper

f = open('/atmosdyn2/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW.txt','rb')
olddata = pickle.load(f)
f.close()

LAT = np.arange(0,90.1,0.4)
for k in olddata.keys():
    for j in olddata[k].keys():
        tmpclat = olddata[k][j]['clat']
        tmpclon = olddata[k][j]['clon']
        olddata[k][j]['clat'] = []
        olddata[k][j]['clon'] = []

        tmpclat2 = np.mean(tmpclat,axis=1).astype(int)
        tmpclon2 = np.mean(tmpclon,axis=1).astype(int)

        for q,r in zip(tmpclon2,tmpclat2):
            CLONIDS, CLATIDS = helper.IFS_radial_ids_correct(200,LAT[r])
            olddata[k][j]['clat'].append(CLATIDS+r)
            olddata[k][j]['clon'].append(CLONIDS+q)
 
f = open('/atmosdyn2/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','wb')
pickle.dump(olddata,f)
f.close()


