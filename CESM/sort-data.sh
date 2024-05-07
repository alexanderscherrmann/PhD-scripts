#!/usr/bin/env bash

in="/net/thermo/atmosdyn2/ascherrmann/015-CESM-WRF/"

cd ${in}

echo $(pwd)
for sea in MAM SON
do
	cd $sea
	echo $(pwd)
	for memb in 1000 1100 1200 1300 0900
	do
		cd ${memb}
		echo $(pwd)

		mkdir 2010 2040 2070 2100
		for k in {1981..2010}
		do
			mv *-$k.txt sum*-$k.nc 2010/
		done
		for k in {2011..2040}
                do
                        mv *-$k.txt sum*-$k.nc 2040/
                done
		for k in {2041..2070}
                do
                        mv *-$k.txt sum*-$k.nc 2070/
                done
		for k in {2071..2100}
                do
                        mv *-$k.txt sum*-$k.nc 2100/
                done
		
		for per in 2010 2040 2070 2100
		do
			cd ${per}
			echo $(pwd)
			rm period-sum-${per}-${sea}.nc
			cdo enssum sum-${sea}-*.nc period-sum-${per}-${sea}.nc
			cdo divc,$(cat total-n-time-steps*.txt) period-sum-${per}-${sea}.nc period-mean-${per}-${sea}.nc
			cd ..
			echo $(pwd)
		done
		cd ..
		echo $(pwd)

	done

	mkdir member-mean
	cd member-mean
	echo $(pwd)
	divc=0
	let "divc = 5 * $(cat ../0900/2010/total-n-time-steps*.txt)"
	for per in 2010 2040 2070 2100
	do
		rm all-member-period-sum-${per}-${sea}.nc
		cdo enssum ../0900/${per}/period-sum-${per}-${sea}.nc ../1000/${per}/period-sum-${per}-${sea}.nc ../1100/${per}/period-sum-${per}-${sea}.nc ../1200/${per}/period-sum-${per}-${sea}.nc ../1300/${per}/period-sum-${per}-${sea}.nc all-member-period-sum-${per}-${sea}.nc
		cdo divc,${divc} all-member-period-sum-${per}-${sea}.nc all-member-period-mean-${per}-${sea}.nc
	done
	cd ..
	echo $(pwd)
	cd ..
	echo $(pwd)
done
