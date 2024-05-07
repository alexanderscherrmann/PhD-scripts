import numpy as np
import pickle
SLP = []
lon = []
lat = []
dates = []
hourstoSLPmin = []
ID = []
savings = ['SLP-','lon-','lat-','ID-','hourstoSLPmin-','dates-']
tmp = dict()
fin = dict()
getids = dict()
getids['initialID'] =np.array([])
getids['newID'] = np.array([])
p = '/home/ascherrmann/009-ERA-5/MED/'

fin = dict()
for x in savings:
    fin[x] = []

for k in range(1979,2021):
  tmp = dict()
  for x in savings:
    f = open(p + x + str(k) + '.txt',"rb")
    tmp[x] = pickle.load(f)
    f.close()

  for l in range(len(tmp['ID-'])):
      if ((len(tmp['ID-'][l])>12) & ((np.max(tmp['SLP-'][l])-np.min(tmp['SLP-'][l]))>5) & (np.min(tmp['SLP-'][l])<1010)):
        dlon = np.array(tmp['lon-'][l][1:])-np.array(tmp['lon-'][l][:-1])
        dlat = np.array(tmp['lat-'][l][1:])-np.array(tmp['lat-'][l][:-1])
        
        r = np.sqrt(dlon**2 + dlat**2)
        if (np.any(r>3.5)):
            uy = np.array(np.where(r>3.5)[0])
            slp = np.where(tmp['SLP-'][l]==np.min(tmp['SLP-'][l]))[0][0]

            if (np.all(uy>slp)):
              for x in savings:
                fin[x].append(tmp[x][l])
            
            elif(len(uy)<3):
                getids['initialID'] = np.append(getids['initialID'],tmp['ID-'][l][0])

                uy = np.insert(uy,0,0)
                for la, lo in enumerate(uy[:-1]):

                    jump = uy[la+1]
                    lth = uy[la+1]-lo
                    if lth>12:
                        idtm = np.ones(lth)*(tmp['ID-'][l][0]+la)
                        lontm = tmp['lon-'][l][lo:jump]
                        lattm = tmp['lat-'][l][lo:jump]
                        slptm = tmp['SLP-'][l][lo:jump]

                        datestm = tmp['dates-'][l][lo:jump]

                        httm = tmp['hourstoSLPmin-'][l][lo:jump]
                        httm = httm - httm[np.where(slptm==np.min(slptm))[0][0]]
                        
                        if(np.max(slptm)-np.min(slptm)>5):
                            fin['lon-'].append(lontm)
                            fin['lat-'].append(lattm)
                            fin['SLP-'].append(slptm)
                            fin['dates-'].append(datestm)
                            fin['hourstoSLPmin-'].append(httm)
                            fin['ID-'].append(idtm)
                            getids['newID'] = np.append(getids['newID'],idtm[0])
                        else:
                            continue
                    else:
                        continue

                    if(uy[la+1]>slp):

                       idtm = np.ones(uy[-1]-lo)*(tmp['ID-'][l][0]+la+1)
                       lontm = tmp['lon-'][l][jump:]
                       lattm = tmp['lat-'][l][jump:]
                       slptm = tmp['SLP-'][l][jump:]
                       datestm = tmp['dates-'][l][jump:]
                       httm = tmp['hourstoSLPmin-'][l][jump:]
                       httm = httm - httm[np.where(slptm==np.min(slptm))[0][0]]

                       if (np.max(slptm)-np.min(slptm)>5) and (np.min(slptm)<1010):
                           fin['lon-'].append(lontm)
                           fin['lat-'].append(lattm)
                           fin['SLP-'].append(slptm)
                           fin['dates-'].append(datestm)
                           fin['hourstoSLPmin-'].append(httm)
                           fin['ID-'].append(idtm)

                           getids['newID'] = np.append(getids['newID'],idtm[0])

                           break
                       else:
                           break

                    else:
                       continue

#        else:
#            for x in savings:
#                fin[x].append(tmp[x][l])

#for x in savings:
#    f = open(p + x + 'furthersel-new.txt',"wb")
#    pickle.dump(fin[x],f)
#    f.close()
    

f = open(p + 'check-IDS.txt','wb')
pickle.dump(getids,f)
f.close()

