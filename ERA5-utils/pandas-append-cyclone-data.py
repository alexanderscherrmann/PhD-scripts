import pickle
import numpy as np
import pandas as pd

clim = np.loadtxt('/home/ascherrmann/009-ERA-5/MED/clim-avPV.txt')

cdata = pd.read_csv('/home/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')
MONTHS = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
MO = np.array([])

for d in cdata['date'].values:
    MO = np.append(MO,MONTHS[int(d[4:6])-1])

cdata['mon'] = MO
df = pd.DataFrame(index=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'],columns=['avPV'])
df['avPV'] = clim

bg = np.array([])
for n,k in zip(cdata['ntraj075'].values,cdata['mon'].values):
        bg = np.append(bg,df['avPV'][k]*n)

cdata['ano'] = cdata['PV075sum']-bg

bg = np.array([])
for n,k in zip(cdata['ntraj'].values,cdata['mon'].values):
    bg = np.append(bg,df['avPV'][k]*n)

cdata['fullano'] = cdata['PVsum']-bg
cdata.to_csv('/home/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv',index=False)
