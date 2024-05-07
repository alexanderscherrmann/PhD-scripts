#!/usr/bin/env bash
in="/net/thermo/atmosdyn2/ascherrmann/015-CESM-WRF/"

cd ${in}
for sea in DJF #MAM SON
do
	cd $sea
	cd member-mean
	divc=0
	let "divc = 5 * $(cat ../0900/2010/total-n-time-steps*.txt)"
	echo $divc
done
