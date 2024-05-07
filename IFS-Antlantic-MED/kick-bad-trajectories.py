import numpy as np
import os

so = 'MED/'
ps = '/home/ascherrmann/010-IFS/ctraj/' + so + 'use/'
pl = '/home/ascherrmann/TT/'# + so

N = 5
traj = np.array([])
H = 48

for d in os.listdir(pl):
    if(d.startswith('trajectories-mature-')):
            traj = np.append(traj,d)

traj = np.sort(traj)

for k in traj[:]:
        f = pl + k
        print(f) 
        d = np.loadtxt(f,skiprows=5)
        
        did = np.where(d[:,3]<0)
        d = np.delete(d,did,axis=0)
        tjstart = np.where(d[:,0]==(-H))[0]
        
        sid = np.array([])
        for n in tjstart:
            sid= np.append(sid,range(n-H,n+1))
        
        sid=sid.astype(int)
        f = ps + k
        np.savetxt(f,d[sid],fmt='%f',delimiter=' ',newline='\n')
    
