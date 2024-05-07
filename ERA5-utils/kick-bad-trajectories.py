import numpy as np
import os

p = '/home/ascherrmann/009-ERA-5/Manos-test/'
p2 = '/home/ascherrmann/009-ERA-5/'
N = 5
traced = np.array([])
traj = np.array([])
H = 47

for d in os.listdir(p):
    if(d.startswith('traced-vars-S-2full')):
            traced = np.append(traced,d)
#    elif(d.startswith('trajectories-mature-2full')):
#            traj = np.append(traj,d)

traced = np.sort(traced)
#traj = np.sort(traj)

for k in range(len(traced)):
#        f = p + traj[k]
        f2 = p + traced[k]
        
#        d = np.loadtxt(f,skiprows=N)
        d = np.loadtxt(f2,skiprows=N)
        
        did = np.where(d[:,3]<0)
        d = np.delete(d,did,axis=0)
#        d2 = np.delete(d2,did,axis=0)
        tjstart = np.where(d[:,0]==(-H))[0]
        
        sid = np.array([])
        for n in tjstart:
            sid= np.append(sid,range(n-H,n+1))
        
        sid=sid.astype(int)
#        f = p2 + traj[k]
        f2 = p2 + traced[k]
        

#        np.savetxt(f,d[sid],fmt='%f',delimiter=' ',newline='\n')
        np.savetxt(f2,d[sid],fmt='%f',delimiter=' ',newline='\n')
    

