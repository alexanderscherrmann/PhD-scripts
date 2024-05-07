import numpy as np
import pandas as pd

p = '/atmosdyn2/ascherrmann/011-all-ERA5/'
df = pd.read_csv(p + 'data/pandas-basic-data-all-deep-over-sea-12h.csv')

MEDend = np.where(df['reg']=='MED')[0][-1] + 1
SAstart = np.where(df['reg']=='SA')[0][0]

df['reg'].values[MEDend:SAstart]='NA'

reg = ['MED','NA','SA','NP','SP','IO']

bd = np.array([-90,-60,-45,-30,-15,0,15,30,45,60,90])
columns = df.columns
for r in reg:
    tmp = df.loc[df['reg']==r]
    lat = tmp['lat'].values
    ID = tmp['ID'].values

    tracks = pd.read_csv(p + 'data/pandas-'+ r +'-track-data-all-deep-over-sea-12h.csv')
    if r!='MED':
     for q,b in enumerate(bd[:-1]):
      
        ### save them distinct for ever region
        ### basic data
        rdf = pd.DataFrame(columns=columns)
        ### track data
        trtmp = pd.DataFrame(columns=list(tracks.columns)[1:])
        # get latitude between boundaries bd to have different regions
        ids = np.where((lat>=b) & (lat<bd[q+1]))[0]
        if len(ids)==0:
            continue
        ### the cyclones should be sorted by minimum SLP and thus select the 1000 deepest 
        ### or if less, as many as there are
        ###
        ids = ids[:2000]
        for i in ids:
            rdf = rdf.append(tmp.loc[tmp['ID']==ID[i]])
            trtmp = trtmp.append(tracks.loc[tracks['ID']==ID[i]][list(tracks.columns)[1:]])

        rdf.to_csv(p + 'data/usecyc/' + r + '-data-deepest-ge-%d-l-%d.csv'%(b,bd[q+1]),index=False)
        trtmp.to_csv(p + 'data/usecyc/' + r + '-tracks-deepest-ge-%d-l-%d.csv'%(b,bd[q+1]),index=False)
    else:
        rdf = tmp[:][:2000]
        trtmp = tracks[list(tracks.columns)[1:]][:2000]
        rdf.to_csv(p + 'data/usecyc/' + r + '-data-deepest.csv',index=False)
        trtmp.to_csv(p + 'data/usecyc/' + r + '-tracks-deepest.csv',index=False)






