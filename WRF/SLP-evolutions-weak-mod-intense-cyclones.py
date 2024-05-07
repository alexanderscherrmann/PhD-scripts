import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from wrf import interplevel as intp
from netCDF4 import Dataset as ds

import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec

import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert

import pickle

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
era5 = '/atmosdyn2/era5/cdf/'

tracks = '/atmosdyn/michaesp/mincl.era-5/tracks/'

which = ['weak-cyclones.csv','moderate-cyclones.csv','intense-cyclones.csv']


seasons = ['DJF','MAM','JJA','SON']
for sea in seasons:
  for wi in which:
    sel = pd.read_csv(ps + sea + '-' + wi)
    for ll in [50,100,150,200]:
      selp = sel.iloc[:ll]
      ID = selp['ID'].values
      dates = selp['dates'].values
      slps = []
      trslp = []
      for i,d in zip(ID,dates):
          y=d[:4]
          m=d[4:6]
          tr = np.loadtxt(tracks + 'fi_' + y + m,skiprows=4)
          if not np.any(tr[:,-1]==i):
              m='%02d'%(int(m)-1)
              if int(m)<1:
                y='%d'%(int(y)-1)
                m='12'
              tr = np.loadtxt(tracks + 'fi_' + y + m,skiprows=4)

          slps.append(tr[np.where(tr[:,-1]==i),3])
          trslp.append(tr[np.where(tr[:,-1]==i),0]
                        -tr[np.where(tr[:,-1]==i),0][0][np.where(slps[-1]==np.min(slps[-1]))[1][-1]])

      fig,ax = plt.subplots()

      
      ax.set_xlabel('time to minSLP [h]')
      ax.set_ylabel('SLP [hPa]')
      ax.set_xlim(-48,48)
      ax.set_ylim(970,1030)
      ax.set_xticks(ticks=np.arange(-48,49,12))
      plt.tick_params(right=True, labelright=False,top=True,labeltop=False)
      for sl,t in zip(slps,trslp):
          ax.plot(np.array(t[0]),np.array(sl[0]),color='k')


      name = 'season/SLP-evol-for-' + wi[:-4] + '-' + sea + '-' + '%d'%ll + '.png'
      fig.savefig(pi + name,dpi=300,bbox_inches='tight')
      plt.close('all')
      
