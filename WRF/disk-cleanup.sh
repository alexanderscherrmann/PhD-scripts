#!bin/bash

cd /atmosdyn2/ascherrmann/013-WRF-sim/
for fol in $(ls -d */)
do
	cd  $fol
	if test -f "wrfout_d01_2000-12-01_03:00:00"
	then
		echo "exists"
		rm wrfout_d01_2000-12-??_03* wrfout_d01_2000-12-??_09* wrfout_d01_2000-12-??_15* wrfout_d01_2000-12-??_21*
	fi
	cd ..
done
