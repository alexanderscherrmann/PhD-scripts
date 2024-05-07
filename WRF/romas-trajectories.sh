#!/bin/bash
# these are 48 hours from mature stage
t=-24
odir=$(pwd)
for k in $(ls -rt trastart-mature-*.txt)
do
	start=$(ls $k | cut -b 17-27)
	name=$(ls $k | cut -b 17-37)
	mon=$(ls traend-$name*.txt | cut -b 30-34)
        nd="/atmosdyn2/ascherrmann/012-WRF-cyclones/trajectories-mature-$name.txt"
	M=$mon

	cp $k /atmosdyn2/ascherrmann/010-IFS/data/$M/
	cd /atmosdyn2/ascherrmann/010-IFS/data/$M/

        /atmosdyn2/ascherrmann/scripts/caltra-IFSORO/prog/caltra $k $t $nd -i 60 -o 60 -ts 5 -ref $start
#	sleep 600s
	rm $k
        cd $odir
done
