#!bin/bash
N=80
W=-150
E=80
S=-20
#for d in 20041221_13 20031224_01 20110126_08 20041212_22 20090102_07 19871209_11 19921229_14
for d in 20031214_01
do
	y=$(echo $d |cut -c 1-4)
	m=$(echo $d |cut -c 5-6)
	da=$(echo $d |cut -c 7-8)
	h=$(echo $d |cut -c 10-11)
	nohup python get-single-data-files-ml-jet.py $N $W $E $S $y $m ${da} $h &
	nohup python get-single-data-files-sf-jet.py $N $W $E $S $y $m ${da} $h &
done
