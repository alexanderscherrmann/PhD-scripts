import numpy as np
import argparse


parser = argparse.ArgumentParser(description="perturb unperturbed reference domain and write new output field")
parser.add_argument('sim',default='',type=str,help='which reference state to perturb: mean or overlap')

args = parser.parse_args()

sim=str(args.sim)
b='/atmosdyn2/ascherrmann/013-WRF-sim/' + sim + '/'

load = np.loadtxt(b + 'startf.xy')
save = np.zeros((len(load[:,0]),4))

save[:,1:] = load

np.savetxt(b+'startf.xy',save,fmt='%.1f',delimiter=' ',newline='\n')

#Reference date 20001201_0000 / Time range       7200 min
#
#  time        x       y     z
#-----------------------------
#

