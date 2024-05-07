#!/usr/bin/env python
import os
import  pandas as pd

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
df = pd.read_csv(ps + 'DJF-intense-cyclones.csv')

when = ['fourdaypriormature','fivedaypriormature','sixdaypriormature','threedaypriormature','twodaypriormature','onedaypriormature','dates']
wq = ['4','5','6']

N=80; W=-150; S=-20; E=80;

clus = 'greece'

for w, we in zip(wq,when):

    df2 = df.iloc[df['region'].values==clus]

    for d in df2[we].values:
        y = d[:4]
        m = d[4:6]
        h = d[-2:]
        day = d[6:8]
        os.system("nohup python /home/ascherrmann/scripts/WRF/get-single-sf-cluster.py %d %d %d %d %s %s %s %s %s %s &"%(N,W,E,S,y,m,day,h,clus,w))
        os.system("nohup python /home/ascherrmann/scripts/WRF/get-single-ml-cluster.py %d %d %d %d %s %s %s %s %s %s &"%(N,W,E,S,y,m,day,h,clus,w))
        os.system("sleep 120")
