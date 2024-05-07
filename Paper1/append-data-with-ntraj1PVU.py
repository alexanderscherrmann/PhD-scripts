import pandas as pd
import numpy as np
import pickle
rdis = 400
pload = '/atmosdyn2/ascherrmann/009-ERA-5/MED/ctraj/use/'
f = open(pload + 'PV-data-dPSP-100-ZB-800-2-%d-correct-distance.txt'%rdis,'rb')
PVdata = pickle.load(f)
f.close()
df = pd.read_csv('/atmosdyn2/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')

datadi = PVdata['rawdata']
ntraj1PVU=np.zeros_like(df['ID'].values)
ntraj125PVU=np.zeros_like(df['ID'].values)
ntraj15PVU=np.zeros_like(df['ID'].values)

for q,ids in enumerate(df['ID'].values):
    ntraj1PVU[q] = np.where(datadi['%06d'%ids]['PV'][:,0]>=1)[0].size
    ntraj125PVU[q] = np.where(datadi['%06d'%ids]['PV'][:,0]>=1.25)[0].size
    ntraj15PVU[q] = np.where(datadi['%06d'%ids]['PV'][:,0]>=1.5)[0].size

df['ntrajgt1PVU'] = ntraj1PVU
df['ntrajgt1.25PVU'] = ntraj125PVU
df['ntrajgt1.5PVU'] = ntraj15PVU

df.to_csv('/atmosdyn2/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv',index=False)
    

