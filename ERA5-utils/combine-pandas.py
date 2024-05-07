import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt

pl = '/home/ascherrmann/009-ERA-5/MED/traj/'
df = pd.read_csv(pl + 'pandas-all-data.csv')

mon = np.array([])
hour = np.array([])
for d in df['date']:
    hour = np.append(hour,int(d[-2:]))
    mon = np.append(mon,int(d[4:6]))

df['month'] = mon
df['hour'] = hour

df = df.loc[df['minSLP']<1000]

col = df.columns

for v in col[4:]:
    for u in col[4:]:
        if v==u:
            continue

        fig,ax = plt.subplots()

        ax.scatter(df[v].values,df[u].values,color='k')
        ax.set_xlabel(v)
        ax.set_ylabel(u)

        fig.savefig('/home/ascherrmann/009-ERA-5/MED/corr/1000-' + v + '-' + u + '-corr.png',dpi=300,bbox_inches='tight')
        plt.close('all')
