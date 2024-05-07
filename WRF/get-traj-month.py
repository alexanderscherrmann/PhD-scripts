import numpy as np
import sys
import os
sys.path.append('/home/ascherrmann/scripts/')
import helper
p = '/atmosdyn2/ascherrmann/012-WRF-cyclones/'
Name = 'traend-'

MONTHS = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
MONTHSN = np.arange(1,13,1)

for d in os.listdir(p):
  if (d.startswith('trastart-mature')):
    date = d[-25:-14]
    ids = int(d[-10:-4])

    yyyy = int(date[:4])
    MM = int(date[4:6])
    DD = int(date[6:8])
    monthid, = np.where(MONTHSN==MM)
    Month = MONTHS[monthid[0]] + date[2:4]

    if ((DD<7) & (ids>120)):
        Month =MONTHS[monthid[0]-1] + '%02d'%(int(date[2:4]))
        if Month=='DEC18':
            Month = 'DEC17'

    MON = Month
    name = Name + d[-25:-4] +'-'+ MON
    np.savetxt(p + name + '.txt',np.zeros(1),fmt='%s',delimiter=' ',newline='\n') 

