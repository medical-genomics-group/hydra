#!/bin/bash

parsingFolder=$1
folder_name=$2
file_name=$3
THIN=$4
path=$5

start_time=$(date -u +%s)

#### START

# count restarts 
wc -l ${path}/${file_name}*.csv | awk '{print $1}' > $parsingFolder/$folder_name/${file_name}_csv.it
rs_done=$(cat $parsingFolder/$folder_name/${file_name}_csv.it | wc -l)

# process files
if [ $rs_done -eq 1 ]
	then
	echo only one .csv file, no restarts

elif [ $rs_done -gt 1 ]
	then
	rs1=''
	rs2='_rs'
	rs3=''
	rs_count=0
	it_done=$(sed '$d' $parsingFolder/$folder_name/${file_name}_csv.it) 
	it_processed=0
	
	for it in $it_done
	do
	
		echo "restart $rs_count"
		
		it_adjusted=$it

		if [ -f "$path/${file_name}${rs2}.csv" ]
		then	
		
		echo "adjusting max it"
		it_adjusted=$(awk '{print $1}' $path/${file_name}${rs2}.csv | sed 's/,//g' | sed -n '1p')
		it_adjusted=$((($it_adjusted/$THIN)))
		it_adjusted=$(($it_adjusted-$it_processed))
	
		fi
	 	
		echo "process 0 : $it_adjusted"
                e1=$(awk '{print $1}' ${path}/${file_name}${rs1}.csv | sed 's/,//g' | sed -n '1p')
                e2=$(awk '{print $1}' ${path}/${file_name}${rs1}.csv | sed 's/,//g' | sed -n $it_adjusted'p')
                echo "process $e1 : $e2" 
			
		sed -n -e "1,${it_adjusted} p"  $path/${file_name}${rs1}.csv >  $parsingFolder/$folder_name/${file_name}${rs1}.tmp.csvLong

		rs1="${rs1}_rs"
		rs2="${rs2}_rs"
		rs_count=$(($rs_count+1))
		it_processed=$(($it_processed+$it_adjusted))

	done

fi		

echo "concat _tmp files"

cat $parsingFolder/$folder_name/${file_name}*.tmp.csvLong > $parsingFolder/$folder_name/${file_name}.csvLong
sort -k1 -n $parsingFolder/$folder_name/${file_name}.csvLong > $parsingFolder/$folder_name/${file_name}.csvLong.sorted
mv $parsingFolder/$folder_name/${file_name}.csvLong.sorted $parsingFolder/$folder_name/${file_name}.csvLong 

echo "remove _tmp files"

rm $parsingFolder/$folder_name/${file_name}*.tmp.csvLong
rm $parsingFolder/$folder_name/${file_name}_csv.it

#### END

end_time=$(date -u +%s)
elapsed=$(($end_time-$start_time))
echo "Total time in sec: $elapsed"

