#!/bin/bash
p=$(pwd)
for k in 19801224_00 19801213_00 19820215_00 19831210_00 19841102_00 19841104_00 19841217_00 19841218_00 19841224_00 19841216_00 19880112_00 19880223_00 19890123_00 19890201_00 19890202_00 19890203_00 19890219_00 19900102_00 19900107_00 19900108_00 19900201_00 19900218_00 19900208_00 19900213_00 19901201_00 19930108_00 19940102_00 19950109_00 19951112_00 19951201_00 19960115_00 19961227_00 19961228_00 19961229_00 19961230_00 19980113_00 19981202_00 19991230_00 20020126_00 20020127_00 20021108_00 20060215_00 20070112_00 20080201_00 20121213_00 20130119_00 20160115_00 20170112_00 20170205_00 20171109_00
do

	y=$(echo $k | cut -b 1-4)
	m=$(echo $k | cut -b 5-6)
	d=$(echo $k | cut -b 7-8)
	h=$(echo $k | cut -b 10-11)

	nohup python get-single-data-files-ml-jet.py 80 -150 80 -20 $y $m $d $h & 
	sleep 180
#	fm=/atmosdyn/era5/cdf/$y/$m/N$k	
#	cdo selvar,MSL,U10M,V10M,SSTK,D2M,T2M $fm /atmosdyn2/ascherrmann/scripts/WRF/tmp/N$k
done

#cdo enssum /atmosdyn2/ascherrmann/scripts/WRF/tmp/N* /atmosdyn2/ascherrmann/scripts/WRF/Nsum-jet-overlap
#rm /atmosdyn2/ascherrmann/scripts/WRF/tmp/N*

#c=50
#cdo divc,$c /atmosdyn2/ascherrmann/scripts/WRF/Nsum-jet-overlap /atmosdyn2/ascherrmann/scripts/WRF/Nmean-jet-overlap
