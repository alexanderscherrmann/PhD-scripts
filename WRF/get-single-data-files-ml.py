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


if (int(m)<8 and int(m)%2==1) or (int(m)>=8 and int(m)%2==0):
    d=31
elif (int(m)==2):
    d=28
    if y%4==0:
        d+=1
else:
    d=30

storefile='data/pl2-%d'%y + m + '.grib'

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type':'reanalysis',
        'variable':[
                'geopotential', 'relative_humidity', 'specific_humidity',
                'temperature', 'u_component_of_wind', 'v_component_of_wind',
        ],
        'pressure_level': [
            '1', '2', '3',
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
            '1000',
        ],
        #               North/West/South/East
        "area":         [N,W,S,E],
        'year':         f'{y}',
        'month':        m,
        'day': ['%02d'%x for x in range(1,d+1)],
        'grid':         [0.5,0.5],
        'time':[
            '12:00',
#		'01:00','02:00',
#            '03:00','04:00','05:00',
#            '06:00','07:00','08:00',
#            '09:00','10:00','11:00',
#            '12:00','13:00','14:00',
#            '15:00','16:00','17:00',
#            '18:00','19:00','20:00',
#            '21:00','22:00','23:00'
        ],
        'format':'grib'
    },
    storefile)


