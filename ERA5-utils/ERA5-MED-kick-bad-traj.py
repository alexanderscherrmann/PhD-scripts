import numpy as np
import os
import shutil

ps = '/home/ascherrmann/009-ERA-5/MED/ctraj/use/'
pl = '/home/ascherrmann/009-ERA-5/MED/ctraj/raw/' 
pl2 = '/home/ascherrmann/009-ERA-5/MED/ctraj/'
pm = '/home/ascherrmann/009-ERA-5/MED/ctraj/cstartdone/'
N = 5
traj = np.array([])
H = 48

already = np.array([])
for d in os.listdir(pl):
    if(d.startswith('trajectories-mature-')):
        if(os.path.isfile(ps + d)):
            continue
        else:
            traj = np.append(traj,d)
#            already = np.append(already,d[-25:])


#for d in os.listdir(pl2):
#    if d.startswith('trastart-mature-'):
#        if np.any(already==d[-25:]):
#            shutil.move(pl2 + d,pm+d)

traj = np.sort(traj)


for k in traj[:]:
        f = pl + k
         
        if (np.loadtxt(f,skiprows=5)[0,0]==0):
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
    
