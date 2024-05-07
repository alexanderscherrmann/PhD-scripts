#!/usr/bin/env python
import cdsapi
import os
import pandas as pd

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
df = pd.read_csv(ps + 'selected-intense-cyclones.csv')

when = ['fourdaypriormature','fivedaypriormature','sixdaypriormature','threedaypriormature','twodaypriormature','onedaypriormature','dates']
wq = ['4','5','6','3','2','1','0']

N=80; W=-150; S=-20; E=80;

c = cdsapi.Client()
for w, we in zip(wq,when):
    sp = 'PV-overlap-data/' + w
    if not os.path.isdir(sp):
        os.mkdir(sp)
    for d in df[we].values:
        y = d[:4]
        m = d[4:6]
        h = d[-2:]
        day = d[6:8]


        storefile = sp + 'sf-overlap' + d[:8] + '_' + d[-2:] + '.grib'

        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                    '2m_temperature', 'mean_sea_level_pressure', 'sea_surface_temperature',
                    'skin_temperature', 'soil_temperature_level_1', 'soil_temperature_level_2',
                    'soil_temperature_level_3', 'soil_temperature_level_4', 'surface_pressure',
                    'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3',
                    'volumetric_soil_water_layer_4',
                ],
        #               North/West/South/East
                "area":         [N,W,S,E],
                'year':         y,
                'month':        m,
                'day':          day,
                'grid':         [0.5,0.5],
                'time':[
                    h + ':00',
                ],
                'format':'grib'
            },
            storefile)


