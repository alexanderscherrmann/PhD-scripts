#!/usr/bin/env python
import argparse
import cdsapi
import os

parser = argparse.ArgumentParser(description=' ')
parser.add_argument('N',default=0,type=int,help='')
parser.add_argument('W',default=0,type=int,help='')
parser.add_argument('E',default=0,type=int,help='')
parser.add_argument('S',default=0,type=int,help='')
parser.add_argument('y',default=0,type=int,help='')
parser.add_argument('m',default='',type=str,help='')


args = parser.parse_args()
N=int(args.N)
W=int(args.W)
E=int(args.E)
S=int(args.S)
y=int(args.y)
m=str(args.m)

if m!='01' and m!='02' and m!='12':
    time=['00:00','12:00',]
else:
    time=['12:00']

if (int(m)<8 and int(m)%2==1) or (int(m)>=8 and int(m)%2==0):
    d=31
elif (int(m)==2):
    d=28
    if y%4==0:
        d+=1
else:
    d=30

storefile='/atmosdyn2/ascherrmann/scripts/WRF/data/sf2-%d'%y + m + '.grib'

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'skin_temperature', 'soil_temperature_level_1', 'soil_temperature_level_2',
            'soil_temperature_level_3', 'soil_temperature_level_4',
            'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3',
            'volumetric_soil_water_layer_4',
        ],
#               North/West/South/East
        "area":         [N,W,S,E],
        'year':         f'{y}',
        'month':        m,
        'day':          ['%02d'%x for x in range(1,d+1)],
        'grid':         [0.5,0.5],
        'time':         time,
        'format':'grib'
    },
    storefile)


