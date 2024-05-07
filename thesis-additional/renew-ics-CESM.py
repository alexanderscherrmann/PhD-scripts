from netCDF4 import Dataset as ds
import numpy as np
import os
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser(description="perturb unperturbed reference domain and write new output field")
parser.add_argument('ref',default='',type=str,help='which reference state to perturb: mean or overlap')
parser.add_argument('per',default='',type=str,help='choose perturbation')

seasons = ['DJF','MAM','SON']
names = ['west','east','south','north']
km=['200','400','800']

ics='/atmosdyn2/ascherrmann/013-WRF-sim/ics/'

seasons = ['DJF','MAM','SON']
period=['2010','2040','2070','2100']
names = ['west','east','south','north']
km=['200','400','800']

ics='/atmosdyn2/ascherrmann/013-WRF-sim/ics/'

for sea in seasons[1:2]:
  for perio in period[:1]:

    os.system('module load dyn_tools;module load conda/2022;source activate iacpy3_2022; bash /atmosdyn2/ascherrmann/scripts/WRF/pvinv-cart/CESM-CESM-automatic-renew.sh %s %s %s %s'%(sea,'max','0',perio))
#   for k in km:
#       for n in names:
#           os.system('module load dyn_tools;module load conda/2022;source activate iacpy3_2022; bash /atmosdyn2/ascherrmann/scripts/WRF/pvinv-cart/CESM-CESM-automatic-renew.sh %s %s %s %s'%(sea,n,k,perio))

os.system('module load dyn_tools; rsync -artv /atmosdyn2/ascherrmann/scripts/WRF/pvinv-cart/perturbed_files-CESM ascherrmann@euler:/cluster/work/climate/ascherrmann/ics/')
