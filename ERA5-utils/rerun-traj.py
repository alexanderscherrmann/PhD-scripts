import numpy as np
import os
import shutil

pl = '/home/ascherrmann/009-ERA-5/MED/traj/' 
ps = '/home/ascherrmann/009-ERA-5/MED/traj/raw/'

cdir = 'used-trastart/'
pm = pl + cdir

for d in os.listdir(pl):
    if(d.startswith('trastart-mature-')):
        name = 'trajectories-mature-' + d[-25:]
        if(os.path.isfile(ps + name)):
            shutil.move(pl + d, pm + d)


