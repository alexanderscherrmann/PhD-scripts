import numpy as np
import wrf
import pickle

load = '/home/ascherrmann/scripts/WRF/isentropic-data/'
f = open(load + 'isentropic-average.txt','rb')
d = pickle.load(f)
f.close()

c = d['counter']
P = d['P']
P[P==0]=np.NaN
mask = np.isnan(P).astype(int)
c[c==0] = np.NaN

P[~(mask.astype(bool))]/=c[~(mask.astype(bool))]
ma = np.ma.array(P,mask=np.isnan(P))

U = d['U']
U[~(mask.astype(bool))]/=c[~(mask.astype(bool))]
V = d['V']
V[~(mask.astype(bool))]/=c[~(mask.astype(bool))]
T = d['T']
T[~(mask.astype(bool))]/=c[~(mask.astype(bool))]
GHT = d['Z']
GHT[~(mask.astype(bool))]/=c[~(mask.astype(bool))]
RH = d['RH']
RH[~(mask.astype(bool))]/=c[~(mask.astype(bool))]


U[mask.astype(bool)]=np.NaN
V[mask.astype(bool)]=np.NaN
T[mask.astype(bool)]=np.NaN
GHT[mask.astype(bool)]=np.NaN
RH[mask.astype(bool)]=np.NaN

### adjust some odd values
P[9,1,376] = (P[9,1,375] + P[9,1,377] + P[9,0,376] + P[9,2,376] )/4
U[9,1,376] = (U[9,1,375] + U[9,1,377] + U[9,0,376] + U[9,2,376] )/4
V[9,1,376] = (V[9,1,375] + V[9,1,377] + V[9,0,376] + V[9,2,376] )/4
T[9,1,376] = (T[9,1,375] + T[9,1,377] + T[9,0,376] + T[9,2,376] )/4
GHT[9,1,376] = (GHT[9,1,375] + GHT[9,1,377] + GHT[9,0,376] + GHT[9,2,376] )/4
RH[9,1,376] = (RH[9,1,375] + RH[9,1,377] + RH[9,0,376] + RH[9,2,376] )/4

P[9,0,373] = (P[9,0,372] + P[9,1,373] + P[9,0,374])/3
U[9,0,373] = (U[9,0,372] + U[9,1,373] + U[9,0,374])/3
V[9,0,373] = (V[9,0,372] + V[9,1,373] + V[9,0,374])/3
RH[9,0,373] = (RH[9,0,372] + RH[9,1,373] + RH[9,0,374])/3
T[9,0,373] = (T[9,0,372] + T[9,1,373] + T[9,0,374])/3
GHT[9,0,373] = (GHT[9,0,372] + GHT[9,1,373] + GHT[9,0,374])/3


pres = np.array([
            '30', '50',
            '70', '100', '125',
            '150', '175', '200',
            '225', '250', '300',
            '350', '400', '450',
            '500', '550', '600'])
pres = pres.astype(int)

LAT = np.linspace(10,80,141)
LON = np.linspace(-150,80,461)

###
### variables on pressure level
### 

up = np.zeros((len(pres),P.shape[1],P.shape[2])) 
vp = np.zeros((len(pres),P.shape[1],P.shape[2]))
tp = np.zeros((len(pres),P.shape[1],P.shape[2]))
rhp = np.zeros((len(pres),P.shape[1],P.shape[2]))
ghtp = np.zeros((len(pres),P.shape[1],P.shape[2]))


###
###
###

lowbord=1000
border=len(LAT)
for k in range(len(P[:,0,0])):
    if len(np.where(np.isnan(P[k:]))[1])==0:
        lowbord=0

    if lowbord==0:
#        print(range(lowbord,border))
        break

#    print(np.unique(np.where(np.isnan(P[k:]))[1]))
    lowbord=np.max(np.unique(np.where(np.isnan(P[k:]))[1])) + 1
    for q,pp in enumerate(pres):
        up[q,lowbord:border,:] = wrf.interplevel(U[k:,lowbord:border,:],P[k:,lowbord:border,:],pp,meta=False)
        ghtp[q,lowbord:border,:] = wrf.interplevel(GHT[k:,lowbord:border,:],P[k:,lowbord:border,:],pp,meta=False)
        rhp[q,lowbord:border,:] = wrf.interplevel(RH[k:,lowbord:border,:],P[k:,lowbord:border,:],pp,meta=False)
        vp[q,lowbord:border,:] = wrf.interplevel(V[k:,lowbord:border,:],P[k:,lowbord:border,:],pp,meta=False)
        tp[q,lowbord:border,:] = wrf.interplevel(T[k:,lowbord:border,:],P[k:,lowbord:border,:],pp,meta=False)

#    print(k,range(lowbord,border))
    border = np.max(np.unique(np.where(np.isnan(P[k:]))[1]))+1

save = dict()
save['U'] = up
save['T'] = tp
save['V'] = vp
save['GHT'] = ghtp
save['RH'] = rhp
save['P'] = pres
save['PS'] = d['PS']/d['2Dcounter']
save['MSL'] = d['MSL']/d['2Dcounter']
save['U10M'] = d['U10M']/d['2Dcounter']
save['V10M'] = d['V10M']/d['2Dcounter']
save['T2M'] = d['T2M']/d['2Dcounter']
save['D2M'] = d['D2M']/d['2Dcounter']
save['SSTK'] = d['SSTK']/d['2Dcounter']
save['lon'] = LON
save['lat'] = LAT


sp = '/home/ascherrmann/scripts/WRF/'
f = open(sp + 'isentropic-average-on-pressure-level-data.txt','wb')
pickle.dump(save,f)
f.close()



