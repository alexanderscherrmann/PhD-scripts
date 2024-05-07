import numpy as np
import pandas as pd
from netCDF4 import Dataset as ds
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/selected-intense-cyclones.csv'
tracks = '/atmosdyn/michaesp/mincl.era-5/tracks/'

sel = pd.read_csv(ps)

STARTLON = np.array([])
STARTLAT = np.array([])
LIFETIME = np.array([])
STARTSLP = np.array([])

for ID,d in zip(sel['ID'].values,sel['dates'].values):
    y = d[:4]
    m = d[4:6]

    tr = np.loadtxt(tracks + 'fi_' + y + m,skiprows=4)
    if not np.any(tr[:,-1]==ID):
          m='%02d'%(int(m)-1)
          if int(m)<1:
            y='%d'%(int(y)-1)
            m='12'
          tr = np.loadtxt(tracks + 'fi_' + y + m,skiprows=4)
    ids = tr[:,-1]
    trid = np.where(ids==ID)[0]
    slps = tr[trid,3]
    time = tr[trid,0]
    lons = tr[trid,1]
    lats = tr[trid,2]

    STARTLON = np.append(STARTLON,lons[0])
    STARTLAT = np.append(STARTLAT,lats[0])
    STARTSLP = np.append(STARTSLP,slps[0])
    LIFETIME = np.append(LIFETIME,len(slps))

sel['startlon'] = STARTLON
sel['startlat'] = STARTLAT
sel['lifetime'] = LIFETIME
sel['startslp'] = STARTSLP
sel.to_csv(ps,index=False)

