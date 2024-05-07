import numpy as np
import os

p = '/home/ascherrmann/009-ERA-5/' + 'MED/cases/'
N = 5
traj = np.array([])
H = 48

for d in os.listdir(p):
    if(d.startswith('trajectories-mature')):
            traj = np.append(traj,d)

traj = np.sort(traj)

for k in range(len(traj)):
        f = p + traj[k]
        d = np.loadtxt(f,skiprows=N)
        
        did = np.where(d[:,3]<0)
        d = np.delete(d,did,axis=0)
        tjstart = np.where(d[:,0]==(-H))[0]
        
        sid = np.array([])
        for n in tjstart:
            sid= np.append(sid,range(n-H,n+1))
        
        sid=sid.astype(int)
        f = p + traj[k]
        np.savetxt(f,d[sid],fmt='%f',delimiter=' ',newline='\n')
    

