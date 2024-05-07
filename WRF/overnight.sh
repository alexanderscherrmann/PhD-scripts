#!bin/bash/
sc=/atmosdyn2/ascherrmann/scripts/WRF/
#python ${sc}4regionsPV-ano.py

dp=/atmosdyn2/ascherrmann/013-WRF-sim/data/4regionsPV/
cd $dp
for k in */;
do
	cd $k/300/
	for d in D*
	do
		nam=$(ls $d|cut -c 2-100)
		cdo -mulc,-1 $d M$nam
	done
	cd ../../
done

cd $sc

nohup bash 4regions.sh &
nohup bash 4regions2.sh &
nohup bash 4regions-ridge.sh &
nohup bash 4regions-ridge2.sh &


