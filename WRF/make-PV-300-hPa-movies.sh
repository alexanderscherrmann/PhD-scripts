for sim in DJF-no-param-1.4-QGPV DJF-no-param-0.7-QGPV DJF-no-param-2.1-QGPV
do
	for t in 01 02 03 04 05 06 07 08 09
	do
		for k in 00 06 12 18
		do	
			python /home/ascherrmann/scripts/WRF/WRF-horizontal-cross-no-wind.py $sim "${t}_${k}" pvo 
		done	

	done
	cd /atmosdyn2/ascherrmann/013-WRF-sim/$sim/
#	bash /home/ascherrmann/scripts/pg-make-movie.sh $sim-PV-300hPa.mp4 1.5 PV-300hPa-2000*.png

done
