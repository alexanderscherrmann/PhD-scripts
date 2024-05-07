import numpy as np
import sys
import os
sys.path.append('/home/ascherrmann/scripts/')
import helper
p = '/home/ascherrmann/009-ERA-5/MED/'

name = 'traend-'
for d in os.listdir(p):
  if (d.startswith('trastart-mature-2full')):
    date = d[21:32]
    yyyy = int(date[0:4])
    MM = int(date[4:6])
    DD = int(date[6:8])
    hh = int(date[9:])
    for k in range(1,49):
        hh-=1
        if (hh<0):
            hh=23
            DD-=1
            if(DD<1):
                MM-=1
                DD=helper.month_days(yyyy)[int(MM)-1]
    Date=str(yyyy)+'%02d%02d_%02d'%(MM,DD,hh)
    np.savetxt(p + name + Date + d[32:-4] +'.txt',np.array([date,Date]),fmt='%s',delimiter=' ',newline='\n') 

