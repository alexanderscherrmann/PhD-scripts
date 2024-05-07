import pandas as pd
import numpy as np
import pickle

regions = ['MED','NA','SA','NP','SP','IO']
mature = dict()
other = dict()
pload = '/atmosdyn2/ascherrmann/011-all-ERA5/data/'
psave =pload

#### raw data
for r in regions:
    f = open(pload + r + '-96h-pre-track-deep-over-sea-12h.txt','rb')
    mature[r] = pickle.load(f)
    f.close()
    f = open(pload + r + '-mature-deep-over-sea-12h.txt','rb')
    other[r] = pickle.load(f)
    f.close()

#### dataframe for basic data
#### region, ID, mature lon and lat, hourstoSLPmin and minimal SLP
df = pd.DataFrame(other['MED'],columns=['ID','lon','lat','htSLPmin','minSLP','dates'])
df['reg'] = 'MED'

cols = list(df.columns)
df = df[[cols[-1]] + cols[:-1]]

for r in regions[1:]:
    tmp = pd.DataFrame(other[r],columns=['ID','lon','lat','htSLPmin','minSLP','dates'])
    tmp['reg'] = r
    tmp = tmp[[cols[-1]] + cols[:-1]]
    df = df.append(tmp,ignore_index=True)
    

df['ID'] = df['ID'].astype('float32').astype('int32')
df['htSLPmin'] = df['htSLPmin'].astype('float32').astype('int32')
df['lon'] = df['lon'].astype('float32')
df['lat'] = df['lat'].astype('float32')
df['minSLP'] = df['minSLP'].astype('float32')

df.to_csv(psave + 'pandas-basic-data-all-deep-over-sea-12h.csv',index=False)

### dataframe for advanced & full data
### advanced includes region, ID and all tracks

###
### creates pandas frame with columns region, ID, -96,-95,.....,-1,0
###

adf = pd.DataFrame(columns=np.append(np.array(['reg','ID']),np.arange(-96,1,1)))
ncol = list(adf.columns)
le = len(mature['MED'][0])
for r in regions:
    ### new dataframe
    adf = pd.DataFrame(columns=np.append(np.array(['reg','ID']),np.arange(-96,1,1)))
    
    ## step through with 2 because even rows are longitude
    ## uneven rows are latitude
    for u in range(0,len(mature[r][:,0]),2):

        nvalues = [r,mature[r][u,0]]
        for k in range(le):
            # save as tuple
            tl = ()
            for l in range(2):
                tl += (mature[r][u+l,k],)
            if k>0:
                nvalues.append(tl)

        adf = adf.append(pd.DataFrame([nvalues],columns=ncol),ignore_index=True)
    adf.to_csv(psave + 'pandas-' + r + '-track-data-all-deep-over-sea-12h.csv',index=False)

