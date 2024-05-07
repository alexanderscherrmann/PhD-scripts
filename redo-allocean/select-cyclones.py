import numpy as np
import pickle
import xarray as xr
savings = ['SLP-','lon-','lat-','ID-','hourstoSLPmin-','dates-','REG-']

p = '/atmosdyn2/ascherrmann/011-all-ERA5/data/'

regions = ['MED','NA','SA','NP','SP','IO']

NAlatup = 83
SB = -83

add = 'all'
add = 'deep-over-sea'

LON = [np.array([-5,42]),#MED
       np.array([-95,15]),#NA
       np.array([-67,30]),#SA
#       np.array([-180,-97]),
       np.array([100,180]),#115- -97#NP
#       np.array([-180,-70]),
       np.array([140,180]),#110- -70#SP
       np.array([20,140])]#IO

LAT = [np.array([28,48]),
        np.array([0,NAlatup]),
        np.array([SB,0]),
        np.array([0,68]),#Npacific
        np.array([SB,0]),#SP
#        np.array([SB,0]),#SP
        np.array([SB,25])]#IO


#LONR = [np.array([-5,2,42]),#MED
#       np.array([-98,-94,-90,-85,-75,-5,15]),#NA
#       np.array([-67,20]),#SA
#       np.array([-180,-98,-94,-85,-76]),#NP
#       np.array([100,180]),#NP
#       np.array([-180,-67]),#SP
#       np.array([120,140,180]),#SP
#       np.array([20,100,120,140])]#IO

#LATR = [np.array([28,42,28,48]),
#        np.array([17,NAlatup,15.5,NAlatup,14,NAlatup,10,NAlatup,0,NAlatup,50,NAlatup]),
#        np.array([SB,0]),
#        np.array([0,68,0,16,0,13,        0,8]),
#        np.array([0,68]),
#        np.array([SB,0]),
#        np.array([-17,0,SB,0]),
#        np.array([SB,25,SB,0,SB,-17])]




NORO = xr.open_dataset('/atmosdyn2/ascherrmann/009-ERA-5/MED/data/NORO')
LSM = NORO['OL'].values[0]

LONG = np.arange(-180,180,0.5)
LATG = np.arange(-90,90.1,0.5)

regtmp = dict()
slpdi = dict()
fulldi = dict()
lon=dict()
lat=dict()
htSLPmin = dict()
dates =dict()
for r in regions:
    regtmp[r] = np.array([])
    slpdi[r] = np.array([])
    lon[r] =np.array([])
    lat[r]= np.array([])
    htSLPmin[r] = np.array([])
    dates[r] = np.array([])


### that loop generates arrays containing all cyclones in that region
for k in range(1979,2021):
  #load SLP,LON,LAT,ID,etc into tmp
  tmp = dict()
  for x in savings:
    f = open(p + x + str(k) + '-' + add + '.txt',"rb")
    tmp[x] = pickle.load(f)
    f.close()

  #loop through all IDs of that year
  for l, ID in enumerate(tmp['ID-']):
      #attribute characteristics to ID in a single dictionary
      # for later
      fulldi[ID] = dict()
      for x in savings:
          fulldi[ID][x] = tmp[x][l]

      ### check in which region it belongs
      for r in regions:
          if tmp['REG-'][l]==r:

              ### removes tracks, where minimum SLP is at the end of the track
              if abs(tmp['hourstoSLPmin-'][l][0])>=len(tmp['lon-'][l]):
                  continue

              ## append corresponding region with mature stage characteristics.
              regtmp[r] = np.append(regtmp[r],ID)
              slpdi[r] = np.append(slpdi[r],np.min(tmp['SLP-'][l]))

              lon[r] = np.append(lon[r],tmp['lon-'][l][abs(tmp['hourstoSLPmin-'][l][0]).astype(int)])
              lat[r] = np.append(lat[r],tmp['lat-'][l][abs(tmp['hourstoSLPmin-'][l][0]).astype(int)])
              htSLPmin[r] = np.append(htSLPmin[r],abs(tmp['hourstoSLPmin-'][l][0]).astype(int))
              dates[r] = np.append(dates[r],tmp['dates-'][l][abs(tmp['hourstoSLPmin-'][l][0]).astype(int)])

reg = dict()
mature = dict()
###track data up to 96 hours prior the mature stage
### cover ID in first column and then t = -96 up to t = 0
hoursel = 12
add = add + '-%dh'%hoursel

for qq,r in enumerate(regions[:]):
    mature[r] = np.array(['ID','lon','lat','htSLPmin','minSLP','dates'])

    ### sort cyclones in each region according to their min SLP
    order = np.argsort(slpdi[r])
    regtmp[r] = regtmp[r][order]
    slpdi[r] = slpdi[r][order]
    lon[r] = lon[r][order]
    lat[r]= lat[r][order]
    htSLPmin[r] = htSLPmin[r][order]
    dates[r]=  dates[r][order]
    ###
    reg[r] = dict()
    
    counter=0
    ### go through all cyclones in that region
    for q in range(0,len(regtmp[r])):
        ### take hoursel instead of -1 * hoursel, as above the absolute value is used
        if htSLPmin[r][q]<hoursel:
            continue
        if r=='NP':
            if ((lon[r][q]>-76 and lon[r][q]<100) or lat[r][q]<0):
                continue

        elif r=='SP':
            if ((lon[r][q]>-67 and lon[r][q]<140) or lat[r][q]>0):
                continue
        else:
            if lon[r][q] > LON[qq][-1] or lon[r][q] < LON[qq][0] or lat[r][q] > LAT[qq][-1] or lat[r][q] < LAT[qq][0]:
                continue


        if lon[r][q]%0.5!=0:
            lon[r][q]= LONG[np.where(abs(LONG-lon[r][q])==np.min(abs(LONG-lon[r][q])))[0][0]]
        if lat[r][q]%0.5!=0:
            lat[r][q]= LATG[np.where(abs(LATG-lat[r][q])==np.min(abs(LATG-lat[r][q])))[0][0]]

        LONID = np.where(LONG==lon[r][q])[0][0]
        LATID = np.where(LATG==lat[r][q])[0][0]
        
        if LSM[LATID,LONID]!=0:
            continue

        ID = regtmp[r][q]
        reg[r][ID] = dict()
        for x in savings:
            reg[r][ID][x] = fulldi[ID][x]

        reg[r][ID]['mlon'] = lon[r][q]
        reg[r][ID]['mlat'] = lat[r][q]
        mature[r] = np.vstack((mature[r],np.array([ID,lon[r][q],lat[r][q],htSLPmin[r][q],slpdi[r][q],dates[r][q]])))

#        lontrack = np.ones(98)*500
#        lattrack = np.ones(98)*500
#        tmplon = fulldi[ID]['lon-'][:abs(htSLPmin[r][q]).astype(int) + 1]
#        tmplat = fulldi[ID]['lat-'][:abs(htSLPmin[r][q]).astype(int) + 1]

#        if len(tmplon)>97:
#            tmplon = np.delete(tmplon,np.arange(0,len(tmplon)-97))
#            tmplat = np.delete(tmplat,np.arange(0,len(tmplat)-97))

#        lontrack[-1*len(tmplon):]=tmplon
#        lattrack[-1*len(tmplat):]=tmplat
#        lontrack[0] = ID
#        lattrack[0] = ID
        
        counter+=1

for r in regions:
    mature[r] = np.delete(mature[r],0,axis=0)

for r in regions[:]:
    f = open(p + r + '-' + add + '.txt',"wb")
    pickle.dump(reg[r],f)
    f.close()

    f = open(p + r + '-mature-' + add + '.txt','wb')
    pickle.dump(mature[r],f)
    f.close()
