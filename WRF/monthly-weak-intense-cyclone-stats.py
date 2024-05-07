import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from wrf import interplevel as intp
from netCDF4 import Dataset as ds
import os
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec

import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert

import pickle

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'

which = ['weak-cyclones.csv','moderate-cyclones.csv','intense-cyclones.csv']
months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

cmap_pv,pv_levels,norm_pv,ticklabels_pv=PV_cmap2()

pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/monthly/stats/'

if not os.path.isdir(pi):
    os.mkdir(pi)

fig,axes = plt.subplots(4,3,sharex=True,sharey=True)
axes = axes.flatten()

for ax,mo in zip(axes,months):
  for co,wi in zip(['blue','red'],which[::2]):
    df=pd.read_csv(ps + mo + '-' + wi)
    ax.hist(df['minSLP'].values,range=[970,1030],bins=24,edgecolor='gray',facecolor=co,alpha=0.7)
  ax.set_ylim(0,30)
  ax.set_xlim(970,1030)
  ax.text(0.025,0.9,mo,color='k',fontsize=8,fontweight='bold',transform=ax.transAxes)


plt.subplots_adjust(wspace=0,hspace=0)
name = 'monthly-SLP-stats.png'
fig.savefig(pi + name,dpi=300,bbox_inches='tight')
plt.close('all')

