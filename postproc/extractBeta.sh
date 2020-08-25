#!/bin/bash

# script to extract nonzero betas from a single run, created by MP
# arguments:
#  parsingFolder - folder where output will be saved
#  folder_name - folder name where the specific run will be stored
#  file_name - name of .bet file for the run
#  THIN -  thinning of the original chain
#  path - path to the files containing the runs.
parsingFolder=$1 
folder_name=$2
file_name=$3
THIN=$4 
path=$5

parsingScriptLocation=./extract_non_zero_betaAll

start_time=$(date -u +%s)

#### START

# count restarts 
wc -l ${path}/${file_name}*.csv | awk '{print $1}' > $parsingFolder/$folder_name/${file_name}_bet.it
rs_done=$(cat $parsingFolder/$folder_name/${file_name}_bet.it | wc -l)

# process files
if [ $rs_done -eq 1 ]
	then
        it_done=$(cat $parsingFolder/$folder_name/${file_name}_bet.it)
	it_done=$(($it_done-1))
	$parsingScriptLocation ${path}/${file_name}.bet 0 $it_done > $parsingFolder/$folder_name/${file_name}_tmp.betLong
        awk '{$1+=0}1' $parsingFolder/$folder_name/${file_name}_tmp.betLong > $parsingFolder/$folder_name/${file_name}.betLong

elif [ $rs_done -gt 1 ]
	then
	rs1=''
	rs2='_rs'
	rs3=''
	rs_count=0
	it_done=$(sed '$d' $parsingFolder/$folder_name/${file_name}_bet.it) 
	it_processed=0
	
	for it in $it_done
	do
	
		echo "restart $rs_count"
		
		it_adjusted=$(($it-1))

		if [ -f "${path}/${file_name}${rs2}.csv" ]
		then	
		
			echo "adjusting max it"
			it_adjusted=$(awk '{print $1}' ${path}/${file_name}${rs2}.csv | sed 's/,//g' | sed -n '1p')

			if [ $rs_count -eq 0 ]
			then
			it_adjusted=$((($it_adjusted)/$THIN))
			it_adjusted=$(($it_adjusted-1))
			fi

			if [ $rs_count -gt 0 ]
			then
			it_adjusted=$((($it_adjusted-5)/$THIN))
                	it_adjusted=$(($it_adjusted-$it_processed))
			fi

		fi	
	 	 
		echo "process 0 : $it_adjusted"
		e1=$(awk '{print $1}' ${path}/${file_name}${rs1}.csv | sed 's/,//g' | sed -n '1p')
		e2=$(awk '{print $1}' ${path}/${file_name}${rs1}.csv | sed 's/,//g' | sed -n $(($it_adjusted+1))'p')
		echo "process $e1 : $e2"
		
		$parsingScriptLocation ${path}/${file_name}${rs1}.bet 0 $it_adjusted > $parsingFolder/$folder_name/${file_name}${rs1}_tmp.betLong

		if [ $rs_count -gt 0 ]
		then
		
		echo "adjust iteration count when rs"
		
		sum_it=$(awk '{print $1}' $parsingFolder/$folder_name/${file_name}${rs3}.betLong | tail -2 | sed -n '1p')
		awk -v s=$(($sum_it+1)) '{print $1+s, $2, $3}' $parsingFolder/$folder_name/${file_name}${rs1}_tmp.betLong > $parsingFolder/$folder_name/${file_name}${rs1}.betLong
		
		rs3="${rs3}_rs"

		elif [ $rs_count -eq 0 ]
		then

		awk '{$1+=0}1' $parsingFolder/$folder_name/${file_name}${rs1}_tmp.betLong > $parsingFolder/$folder_name/${file_name}${rs1}.betLong
		
		fi
		
		rs1="${rs1}_rs"
		rs2="${rs2}_rs"
		rs_count=$(($rs_count+1))
		it_processed=$(($it_processed+$it_adjusted+1))

	done

fi		


echo "remove _tmp files"

rm $parsingFolder/$folder_name/${file_name}*tmp.betLong
rm $parsingFolder/$folder_name/${file_name}_bet.it

echo "concat _rs files"

cat $parsingFolder/$folder_name/${file_name}*.betLong > $parsingFolder/$folder_name/${file_name}.all.betLong
mv $parsingFolder/$folder_name/${file_name}.all.betLong $parsingFolder/$folder_name/${file_name}.betLong
rm $parsingFolder/$folder_name/${file_name}_*.betLong


#### END

end_time=$(date -u +%s)
elapsed=$(($end_time-$start_time))
echo "Total time in sec: $elapsed"

