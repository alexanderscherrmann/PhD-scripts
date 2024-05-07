#!/bin/bash

pl=ETA-15-pl.grib
sf=ETA-15-sf.grib
l=$(cdo -ntime $pl)
for s in $(seq 1 1 $l)
do
	echo $s
	cdo seltimestep,$s $pl tmppl.grib2
	cdo seltimestep,$s $sf tmpsf.grib2
	sleep 1
	date=$(cdo showdate tmpsf.grib2 |cut -b 3-6,8-9,11-12)
	hour=$(cdo showtime tmpsf.grib2 |cut -b 2-3)
	min=$(cdo showtime tmpsf.grib2 |cut -b 5-6)
	mv tmppl.grib2 pl\_$date\_$hour\_$min.grib2
	mv tmpsf.grib2 sf\_$date\_$hour\_$min.grib2

done
