import re
import numpy as np
re.findall(r"[-+]?(?:\d*\.\d+|\d+)", "5-QG-0.5")
import os
for d in os.listdir('/home/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'):
    if d.endswith('-FILTER'):
        strings=np.array([])
        for l in re.findall(r"[-+]?(?:\d*\.\d+|\d+)",d):
            strings = np.append(strings,float(l[1:]))
        if strings.size==2 and strings[0]<300:
            strings = np.append(200.,strings)
        if strings.size==2 and strings[0]==300:
            strings = np.append(strings,0.0)

        print(strings) 
