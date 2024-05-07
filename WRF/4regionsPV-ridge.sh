#!bin/bash
module load netcdf
h=$(pwd)
sc="/home/ascherrmann/scripts/WRF/"
ed="/atmosdyn2/ascherrmann/013-WRF-sim/data/4regionsPV/"
pr=(300) #(250 300 350 400 450)
#PV=(2) #(2 2 1.5 1 1)
cd $ed
#fix="D"
#fil=streamer-mask.nc
fix="M"
PV=(1)
fil=ridge-mask.nc
for k in $(ls -d */ |head -378)
do
	echo $k
	cd $k
	for i in ${!pr[@]}
	do
		cd ${pr[i]}
		idd=$(echo $k|cut -c 1-6)
		eny=$(ls  D* |tail -1 | cut -c 2-5)
		enm=$(ls  D* |tail -1 | cut -c 6-7 | bc)
		endd=$(ls D* |tail -1 | cut -c 8-9 | bc)
		enh=$(ls  D* |tail -1 | cut -c 11-12 | bc)
		
		sy=$(ls D* |head -1 | cut -c 2-5)
		sm=$(ls D* |head -1 | cut -c 6-7 | bc)
		sd=$(ls D* |head -1 | cut -c 8-9 | bc)
		sh=$(ls D* |head -1 | cut -c 11-12 | bc)
		
		echo $idd $eny $enm $endd $enh $sy $sm $sd $sh
		echo $(ls M*)
		
		cp ${sc}idpv.F90 ${sc}idr.F90
		
		sed -i "s/XXSTARTYXX/${sy}/g" ${sc}idr.F90
		sed -i "s/XXSTARTMXX/${sm}/g" ${sc}idr.F90
		sed -i "s/XXSTARTDXX/${sd}/g" ${sc}idr.F90
		sed -i "s/XXSTARTHXX/${sh}/g" ${sc}idr.F90
		
		
		sed -i "s/XXENDYXX/${eny}/g" ${sc}idr.F90
		sed -i "s/XXENDMXX/${enm}/g" ${sc}idr.F90
		sed -i "s/XXENDDXX/${endd}/g" ${sc}idr.F90
		sed -i "s/XXENDHXX/${enh}/g" ${sc}idr.F90
		sed -i "s/XXIDXX/${idd}/g" ${sc}idr.F90
		sed -i "s/XXPRXX/${pr[i]}/g" ${sc}idr.F90
		sed -i "s/XXTHRESHXX/${PV[i]}/g" ${sc}idr.F90
		sed -i "s/XXFILEXX/${fix}/g" ${sc}idr.F90
		sed -i "s/XXOUTNAMEXX/${fil}/g" ${sc}idr.F90

		if [ -f "$fil" ]
		then
			#rm $fil
		        #gfortran ${sc}idr.F90 -o oler `nf-config --fflags --flibs`
			#./oler	
			cd ../
			continue
		
		else
			gfortran ${sc}idr.F90 -o oler `nf-config --fflags --flibs`
			./oler
		fi
		cd ../
	done
	cd ../
done
