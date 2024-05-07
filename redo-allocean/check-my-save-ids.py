import numpy as np
import pickle
savings = ['SLP-','lon-','lat-','ID-','hourstoSLPmin-','dates-','REG-']

p = '/atmosdyn2/ascherrmann/011-all-ERA5/data/'
regions = ['MED','NA','SA','NP','SP','IO']
add = 'deep-over-sea'

ids = np.array([])
### that loop generates arrays containing all cyclones in that region
for k in range(1979,2021):
  #load SLP,LON,LAT,ID,etc into tmp
  tmp = dict()
  for x in savings:
    if x!='ID-':
        continue
    f = open(p + x + str(k) + '-' + add + '.txt',"rb")
    tmp[x] = pickle.load(f)
    f.close()

    

    a,c = np.unique(np.array(tmp['ID-']),return_counts=True)
    print(k,np.where(c>1)[0].size)

    ids = np.append(ids,np.array(tmp['ID-']))

a,c = np.unique(ids,return_counts=True)
print(np.where(c>1)[0].size)
