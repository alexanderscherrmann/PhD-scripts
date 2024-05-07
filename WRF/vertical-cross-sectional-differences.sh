#!/bin/bash
i=/home/ascherrmann/scripts/WRF/
for m in 12
do
	p=/atmosdyn/era5/cdf/2000/$m/
	cd $p
	for k in $(ls S*_00 |cut -c 2-100)
	do	
		for lo in -120 -90 -60 -30 0 30 60
		do
			nohup python $i/vcross-wrf-compare.py $k $lo &
			sleep 20
		done
	done
done
