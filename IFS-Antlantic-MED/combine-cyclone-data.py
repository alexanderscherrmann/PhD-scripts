import pickle
import numpy as np
MONTHS = np.array(['DEC17','JAN18','FEB18','MAR18','APR18','MAY18','JUN18','JUL18','AUG18','SEP18','OCT18','NOV18'])
f = open('DEC17/All-CYC-entire-year-NEWDEC17-correct.txt','rb')
d = pickle.load(f)
f.close()
for k in MONTHS[1:]:
    f = open(k + '/All-CYC-entire-year-NEW'+ k + '-correct.txt','rb')
    tmp = pickle.load(f)
    f.close()
    d[k] = tmp[k]
f = open('All-CYC-entire-year-NEW-correct.txt','wb')
pickle.dump(d,f)
f.close()

