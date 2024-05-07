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
parser.add_argument('d',default='',type=str,help='')
parser.add_argument('h',default='',type=str,help='')


args = parser.parse_args()
N=int(args.N)
W=int(args.W)
E=int(args.E)
S=int(args.S)
y=int(args.y)
m=str(args.m)
d=str(args.d)
h=str(args.h)

storefile='test-overlap-ridge/pl-overlap%d'%y + m  + d + '_' + h + '.grib'

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type':'reanalysis',
        'variable':[
                'geopotential', 'relative_humidity', 'specific_humidity',
                'temperature', 'u_component_of_wind', 'v_component_of_wind','potential_vorticity'
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
        'day':          d,
        'grid':         [0.5,0.5],
        'time':[
            h + ':00',
        ],
        'format':'grib'
    },
    storefile)


