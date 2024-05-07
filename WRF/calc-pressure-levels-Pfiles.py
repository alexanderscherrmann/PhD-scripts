import numpy as np
import pickle
import xarray as xr
import wrf

LON = np.round(np.linspace(-180,179.5,720),1)
LAT = np.round(np.linspace(-90,90,361),1)

fs = ['Pmean','P-jet-overlap']
pres = np.array(['1', '2', '3',
            '5', '7', '10',
            '20', '30', '50',
            '70', '100', '125',
            '150', '175', '200',
            '225', '250', '300',
            '350', '400', '450',
            '500', '550', '600',
            '650', '700', '750',
            '775', '800', '825',
            '850', '875', '900',
            '925', '950', '975',
            '1000'])
pres = pres.astype(int)
## save variables

for f in fs:
  p = xr.open_dataset('/home/ascherrmann/scripts/WRF/' + f)
  PS = p.PS.values[0]
  u = p.U.values[0]
  v = p.V.values[0]
  q = p.Q.values[0]
  t = p.T.values[0]
  ak = p.hyam.values[137-98:]
  bk = p.hybm.values[137-98:]

  p3d = np.tile(PS,(u.shape[0],1,1))
  p3d = (ak/100 + bk * p3d.T).T

  np.savetxt('/home/ascherrmann/scripts/WRF/'+f+'-surface-pressure.txt',PS,fmt='%f',delimiter=' ',newline='\n')
    
  U = wrf.interplevel(u,p3d,pres,meta=False)
  V = wrf.interplevel(v,p3d,pres,meta=False)
  T = wrf.interplevel(t,p3d,pres,meta=False)
  Q = wrf.interplevel(q,p3d,pres,meta=False)

  save = dict()
  save['U'] = U.data
  save['V'] = V.data
  save['T'] = T.data
  save['Q'] = Q.data
  save['P'] = pres

  fsa = open('/home/ascherrmann/scripts/WRF/'+f+'-pressure-lvl.txt','wb')
  pickle.dump(save,fsa)
  fsa.close()
