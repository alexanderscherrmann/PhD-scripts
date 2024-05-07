from netCDF4 import Dataset as ds
import numpy as np
import os
import pandas as pd
import pickle 

import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper

minlon = -10
minlat = 25
maxlat = 60
maxlon = 45

minrlon = -70
minrlat = 30
maxrlat = 75
maxrlon = 10

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

lonsr = np.where((LON>=minrlon) & (LON<=maxrlon))[0]
latsr = np.where((LAT<=maxrlat) & (LAT>=minrlat))[0]

lor0,lor1,lar0,lar1 = lonsr[0],lonsr[-1]+1,latsr[0],latsr[-1]+1

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/'
pc = '/atmosdyn/michaesp/mincl.era-5/cdf.final/'

overlap = dict()

streamertypes = dict()
whichoverlap = dict()
le = 0
checkval = 0.8
minn = 300
prelen=0
newtype=1
prelen2 = 0
iterations = 0

sizecheck=0.5

mth=3
lth=57
gth=50

while le<400:
    
    ### hard break for initial test
    if iterations>=400:
        print('broke by iterations')
        break

    iterations+=1

    ### loop throught cases and their streamers
    ### first iteration is only the 325832 case
    for dirs in os.listdir(ps):
      if dirs[-1]=='c' or dirs[-1]=='t':
          continue
    

      ### if type already captured continue
      ### in directory loop
      ###
      con = 0
      for ke in streamertypes.keys():
          if np.any(np.array(streamertypes[ke])==int(dirs)):
             con+=1
      
      if con!=0:
          continue
      
      ### if no more streamers overlap with current streamer type create a new one
      if newtype==1:
          tmpmature = np.loadtxt(ps + dirs + '/overlapping-streamer-tracks-300.txt',skiprows=1)
          
          ### if empty or no overlap at mature stge skip this one
          if tmpmature.size==0:
              continue

          tmpmature = tmpmature.reshape(-1,10)
          ### if there is no streamer overlapping with the cyclone mask at the mature stage continue
          if not np.any(tmpmature[:,0]==0):
              continue
          
          ###
          ### streamer id is selected from streamer overlapping with cyclone mask at the mature stage
          ###

          mat = np.where(tmpmature[:,0]==0)[0]
          areaovlp = tmpmature[mat,2]
          tmpstreamerid = tmpmature[mat[np.argmax(areaovlp)],1]
          ### define new type of streamer to check for overlap
          ### adjust according to streamer that overlaps with cyclone
          ### and largest ridge in that area
          
          streamertypes[dirs] = [int(dirs)]
          refr = ds(ps + dirs + '/300/ridge-mask.nc','r')
          refs = ds(ps + dirs + '/300/streamer-mask.nc','r')
          refr = refr.variables['mask'][56,lar0:lar1,lor0:lor1]
          refs = refs.variables['mask'][56,la0:la1,lo0:lo1]

          currenttype=dirs
          
          #### new streamer and ridge masks
          #### slabel and rlabel
          ####

          slabel = tmpstreamerid
          rlabels = np.array([])
          rlens = np.array([])
          for v in np.unique(refr)[1:]:
            rlens = np.append(rlens,len(np.where(refr==v)[0]))
            rlabels = np.append(rlabels,v)

          
          ### new lengths of streamer and ridge masks
          ### select largest ridge in that area
          ###

          rrl = len(np.where(refr==rlabels[np.argmax(rlens)])[0])
          rsl = len(np.where(refs==slabel)[0])

          rlabel = rlabels[np.argmax(rlens)]
          whichoverlap[dirs] = [(int(dirs),56,slabel,rlabel)]
          ### now check the newly set streamer and ridge pattern
          newtype=0
          print('new streamer id, new ridge id')
          print(slabel,rlabel)

      
      if dirs[-1]!='c' and dirs[-1]!='t':    
       ID = dirs +'/'
       overlap[int(dirs)] = dict()
    
       for q,pr in enumerate(['300']):
        overlap[int(dirs)][pr] = dict()
        overlap[int(dirs)][pr]['streamer'] = []
        overlap[int(dirs)][pr]['ridge'] = []
    
        d = ds(ps + ID + '%s/streamer-mask.nc'%pr,'r')
        r = ds(ps + ID + '%s/ridge-mask.nc'%pr,'r')
    
        mask = d.variables['mask'][:,la0:la1,lo0:lo1]
        rmask = r.variables['mask'][:,lar0:lar1,lor0:lor1]
        
        prvl = 0
        ### for each timestep of each cyclone that has not yet been captured
        ### check for overlapping streamers
        ###
        for k in range(gth,lth+1):
            for v in np.unique(mask[k])[1:]:
                if v==0:
                    continue
                sov = np.zeros_like(refs)
                sov[refs==slabel] +=1
    
                sov[mask[k]==v]+=1
                checklen = len(np.where(mask[k]==v)[0])
                if np.any(sov==2):
                  ## check for similar size
                  if rsl/checklen > sizecheck and checklen/rsl>sizecheck:
                    sl = len(np.where(mask[k]==v)[0])
                    if sl<checkval:
                        continue

                    if sl>=rsl:
                        if len(refs[np.where(sov==2)])/rsl>checkval:
                          overlap[int(dirs)][pr]['streamer'].append((int(currenttype),k,v))

                    else:
                        if len(mask[k][np.where(sov==2)])/sl>checkval:
                          overlap[int(dirs)][pr]['streamer'].append((int(currenttype),k,v))
    
            ### if no streamer overlaps
            ### or no new is added
            if len(overlap[int(dirs)][pr]['streamer'])==0 or len(overlap[int(dirs)][pr]['streamer'])==prvl:
                continue
            
            ### if the current time step has no overlapping streamer
            if overlap[int(dirs)][pr]['streamer'][-1][1]!=k:
                continue
    
            ### 
            ### 
            ### 

            for v in np.unique(rmask[k][1:]):
                if v==0:
                    continue

                rov = np.zeros_like(refr)
                rov[refr==rlabel] +=1
                rov[rmask[k]==v]+=1
                checklen = len(np.where(rmask[k]==v)[0])
                if np.any(rov==2):
                  if rrl/checklen > sizecheck and checklen/rrl>sizecheck:
                    rl = len(np.where(rmask[k]==v)[0])
                    if rl<checkval:
                        continue

                    if rl>=rrl:
                        if len(refr[np.where(rov==2)])/rrl>checkval:
                            overlap[int(dirs)][pr]['ridge'].append((int(currenttype),k,v))
                    else:
                        if len(rmask[k][np.where(rov==2)])/rl>checkval:
                            overlap[int(dirs)][pr]['ridge'].append((int(currenttype),k,v))
            
            prvl = len(overlap[int(dirs)][pr]['streamer'])
       
        ###
        ### if ridge time and streamer time dont overlap continue
        ### remove those entries
        ###

        rvls = np.array([])
        for rvl in overlap[int(dirs)][pr]['ridge']:
            rvls = np.append(rvls,rvl[1])
        rvls = np.unique(rvls)
    
        rmvl = []
        for svl in overlap[int(dirs)][pr]['streamer']:
            if not np.any(rvls==svl[1]):
                rmvl.append(svl)
    
        for rmv in rmvl:
            overlap[int(dirs)][pr]['streamer'].remove(rmv)

       if len(overlap[int(dirs)][pr]['streamer'])!=0 and len(overlap[int(dirs)][pr]['ridge'])!=0:
           ok = 0
           for st in overlap[int(dirs)][pr]['streamer']:
               if st[1]>gth and st[1]<lth:
                   ok+=1
           if ok>mth:
            streamertypes[currenttype].append(int(dirs))

    le=0
    for ke in streamertypes.keys():
        le+=len(streamertypes[ke])
   
    if le==prelen:
        newtype=1

    print(le)
    prelen=le


f = open(ps + 'test-mature-size-%.1f-ridge-streamer-types-%.1f-%d-more-than-%d-in-%d-%d.txt'%(sizecheck,checkval,minn,mth,gth,lth),'wb')
pickle.dump(overlap,f)
f.close()


f = open(ps + 'test-mature-size-%.1f-streamer-types-%.1f-%d-more-than-%d-in-%d-%d.txt'%(sizecheck,checkval,minn,mth,gth,lth),'wb')
pickle.dump(streamertypes,f)
f.close()
