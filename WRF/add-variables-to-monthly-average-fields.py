# coding: utf-8
import pickle
months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
f = open('monthly-average-fields.txt','rb')
gd = pickle.load(f)
f.close()
vd = ['Q850','THE850','Pat1.5PVU','Pat2PVU']
for mo in months:
    f = open(mo + '-average-fields2.txt','rb')
    nd = pickle.load(f)
    f.close()
    f = open(mo + '-average-fields.txt','rb')
    od = pickle.load(f)
    f.close()
    for wi in ['weak-cyclones.csv','moderate-cyclones.csv','intense-cyclones.csv']:
        for ll in [50]:
            for we in when = ['fourdaypriormature','fivedaypriormature','sixdaypriormature','sevendaypriormature','threedaypriormature','twodaypriormature','onedaypriormature','dates']:
                for nv in vd:
                    gd[mo][wi][ll][we][nv] = nd[mo][wi][ll][we][nv]
for mo in months:
    f = open(mo + '-average-fields2.txt','rb')
    nd = pickle.load(f)
    f.close()
    f = open(mo + '-average-fields.txt','rb')
    od = pickle.load(f)
    f.close()
    for wi in ['weak-cyclones.csv','moderate-cyclones.csv','intense-cyclones.csv']:
        for ll in [50]:
            for we in ['fourdaypriormature','fivedaypriormature','sixdaypriormature','sevendaypriormature','threedaypriormature','twodaypriormature','onedaypriormature','dates']:
                for nv in vd:
                    gd[mo][wi][ll][we][nv] = nd[mo][wi][ll][we][nv]
                    od[mo][wi][ll][we][nv] = nd[mo][wi][ll][we][nv]
    f = open(mo + '-average-fields.txt','wb')
    pickle.dump(od,f)
    f.close()
    
gd[mo][wi][ll][we].keys()
gd[mo][wi][ll][we]['PV300hPa']
gd[mo][wi][ll][we]['THE850']
gd[mo][wi][ll][we]['wcbcounter']
gd[mo][wi][ll][we]['Pat2PVU']
f = open('monthly-average-fields.txt','wb')
pickle.dump(gd,f)
f.close()
%save -r /home/ascherrmann/scripts/WRF/add-variables-to-monthly-average-fields.py 1-999
