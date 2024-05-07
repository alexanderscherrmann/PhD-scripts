#!/bin/bash
for y in {1979..2020..1}
do
	for m in 01 02 12
	do
		for d in $(ls /atmosdyn/era5/cdf/$y/$m/P*_00)
		do
			p=$(ls $d | cut -c 1-27)
			da=$(ls $d | cut -c 29-1000)

			ln -sf $(ls ${p}*${da}) .
			
			###
			### run geopot calc

			nohup ./geopot2p P${da} H${da} G${da} &
			sleep 27

			### now calc variables on isentropes
			###

			nohup python ../gather-ic-data-on-isentropic-levels.py ${da}
			sleep 10
			rm *${da}
		done
	done
done
