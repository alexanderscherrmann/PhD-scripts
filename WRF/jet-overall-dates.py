import numpy as np
import xarray as xr
import os
import pickle

pl = '/net/helium/atmosdyn/erainterim/clim/jet/cdf/'
tmp = xr.open_dataset(pl + '2005/12/J20051207_00')
rm = tmp.dljet.values[0,0,120:,80:171]
lorm, larm = np.where(rm==1)

c = len(lorm)

dates = np.array([])
perc = np.array([])
for y in range(1979,2018):
    for m in np.array([1,2,11,12]):
        for f in os.listdir(pl + '%d/%02d/'%(y,m)):
          if f.startswith('J'):
            tmp = xr.open_dataset(pl + '%d/%02d/'%(y,m) + f)
            rm = tmp.dljet.values[0,0,120:,30:151]

            co = np.sum(rm[lorm,larm])
            if co/c>0.5:

                dates = np.append(dates,f[-11:])
                perc = np.append(perc,co/c)

save = dict()
save['date'] = dates
save['perc'] = perc

f = open('/home/ascherrmann/scripts/WRF/jet-overlap.txt','wb')
pickle.dump(save,f)
f.close()

