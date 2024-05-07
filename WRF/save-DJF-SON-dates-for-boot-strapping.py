import numpy as np
import os

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
pd = '/atmosdyn2/ascherrmann/009-ERA-5/MED/data/'

datesSON = np.array([])
datesDJF = np.array([])

for f in os.listdir(pd):
    if f[5:7]=='01' or f[5:7]=='02' or f[5:7]=='12':
        datesDJF = np.append(datesDJF,f[1:])

    elif f[5:7]=='10' or f[5:7]=='11' or f[5:7]=='09':
        datesSON = np.append(datesSON,f[1:])

    else:
        continue

np.savetxt(ps + 'draw-dates-from-DJF.txt',datesDJF,fmt='%s',delimiter=' ',newline='\n')
np.savetxt(ps + 'draw-dates-from-SON.txt',datesSON,fmt='%s',delimiter=' ',newline='\n')

