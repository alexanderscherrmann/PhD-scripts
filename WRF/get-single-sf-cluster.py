#!/usr/bin/env python
import argparse
import cdsapi
import os

parser = argparse.ArgumentParser(description=' ')

parser.add_argument('N',default=0,type=int,help='')
parser.add_argument('W',default=0,type=int,help='')
parser.add_argument('E',default=0,type=int,help='')
parser.add_argument('S',default=0,type=int,help='')
parser.add_argument('y',default=0,type=str,help='')
parser.add_argument('m',default='',type=str,help='')
parser.add_argument('d',default='',type=str,help='')
parser.add_argument('h',default='',type=str,help='')

parser.add_argument('cluster',default='',type=str,help='')
parser.add_argument('when',default='',type=str,help='')

args = parser.parse_args()
N=int(args.N)
W=int(args.W) 
E=int(args.E)
S=int(args.S)
y=str(args.y)
m=str(args.m)
d=str(args.d)
h=str(args.h)
w=str(args.when)
clus = str(args.cluster)

storefile='cluster-overlap/' + clus + '/' + w + '/sf' + y + m + d + '_' + h +  '.grib'

if not os.path.isdir('cluster-overlap/'):
    os.mkdir('cluster-overlap/')
if not os.path.isdir('cluster-overlap/' + clus):
    os.mkdir('cluster-overlap/' + clus)
if not os.path.isdir('cluster-overlap/' + clus + '/' + w):
    os.mkdir('cluster-overlap/' + clus + '/' + w)

c = cdsapi.Client()
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
        'day':          d,
        'grid':         [0.5,0.5],
        'time':         [h + ':00',],
        'format':'grib'
    },
    storefile)


