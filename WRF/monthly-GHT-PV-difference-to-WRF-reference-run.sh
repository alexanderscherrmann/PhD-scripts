#!/bin/bash
i=/atmosdyn2/ascherrmann/scripts/WRF/
for m in 12
do
	p=/atmosdyn/era5/cdf/2000/$m
	cd $p
	for k in $(ls S*_00 |cut -c 2-100)
	do	
		nohup python $i/hrzt-wrf-day-anomaly.py $k &
		sleep 20
	done
done
