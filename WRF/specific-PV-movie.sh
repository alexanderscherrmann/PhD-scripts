#!bin/bash
swr=/home/ascherrmann/scripts/WRF/
#for sim in DJF-clim-max-U-at-300-hPa-0.3-QGPV DJF-clim-max-U-at-300-hPa-0.5-QGPV DJF-clim-max-U-at-300-hPa-0.7-QGPV DJF-clim-max-U-at-300-hPa-0.9-QGPV DJF-clim-max-U-at-300-hPa-1.1-QGPV DJF-clim-max-U-at-300-hPa-1.4-QGPV DJF-clim-max-U-at-300-hPa-1.7-QGPV DJF-clim-max-U-at-300-hPa-2.1-QGPV DJF-clim-max-U-at-300-hPa-2.8-QGPV
for sim in DJF-200-km-east-from-max-300-hPa-2.1-QGPV DJF-200-km-south-from-max-300-hPa-2.1-QGPV DJF-200-km-north-from-max-300-hPa-2.1-QGPV DJF-200-km-west-from-max-300-hPa-2.1-QGPV DJF-400-km-east-from-max-300-hPa-2.1-QGPV 	DJF-400-km-west-from-max-300-hPa-2.1-QGPV DJF-400-km-north-from-max-300-hPa-2.1-QGPV DJF-400-km-south-from-max-300-hPa-2.1-QGPV
do 
	for d in 01 02 03 04 05 06 07 08 09 10
       	do
	       	for h in 00 06 12 18
	       	do
		       	python WRF-horizontal-cross-no-wind.py $sim "${d}_${h}" pvo
	       	done
       	done
       	cd /atmosdyn2/ascherrmann/013-WRF-sim/$sim
       	bash /home/ascherrmann/scripts/pg-make-movie.sh PV-$sim.mp4 1.5 PV-300hPa-*.png
       	cd $swr
done

