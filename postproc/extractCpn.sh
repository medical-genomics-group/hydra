#!/bin/bash
#SBATCH --job-name=extrCpn
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time 0-03:00:00
#SBATCH --partition=parallel
#SBATCH --account=ext-unil-ctgg

parsingFolder=$1
folder_name=$2
file_name=$3
THIN=$4
path=$5

parsingScriptLocation=/work/ext-unil-ctgg/common_software/PostProcessing/extract_non_zero_cpnAll

start_time=$(date -u +%s)

#### START 

# count restarts 
wc -l ${path}/${file_name}*.csv | awk '{print $1}' > $parsingFolder/$folder_name/${file_name}_cpn.it
rs_done=$(cat $parsingFolder/$folder_name/${file_name}_cpn.it | wc -l)

# process files
if [ $rs_done -eq 1 ]
	then
        it_done=$(cat $parsingFolder/$folder_name/${file_name}_cpn.it)
	it_done=$(($it_done-1))
	$parsingScriptLocation ${path}/${file_name}.cpn ${path}/${file_name}.bet 0 $it_done > $parsingFolder/$folder_name/${file_name}_tmp.cpnLong
        awk '{$1+=0}1' $parsingFolder/$folder_name/${file_name}_tmp.cpnLong > $parsingFolder/$folder_name/${file_name}.cpnLong

elif [ $rs_done -gt 1 ]
	then
	rs1=''
	rs2='_rs'
	rs3=''
	rs_count=0
	it_done=$(sed '$d' $parsingFolder/$folder_name/${file_name}_cpn.it)
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
                e2=$(awk '{print $1}' ${path}/${file_name}${rs1}.csv | sed 's/,//g' | sed -n $((it_adjusted+1))'p')
                echo "process $e1 : $e2"
	
		$parsingScriptLocation ${path}/${file_name}${rs1}.cpn ${path}/${file_name}${rs1}.bet 0 $it_adjusted > $parsingFolder/$folder_name/${file_name}${rs1}_tmp.cpnLong

		if [ $rs_count -gt 0 ]
		then
		
		echo "adjust iteration count when rs"
		
		sum_it=$(awk '{print $1}' $parsingFolder/$folder_name/${file_name}${rs3}.cpnLong | tail -2 | sed -n '1p')
		awk -v s=$(($sum_it+1)) '{print $1+s, $2, $3}' $parsingFolder/$folder_name/${file_name}${rs1}_tmp.cpnLong > $parsingFolder/$folder_name/${file_name}${rs1}.cpnLong
		
		rs3="${rs3}_rs"

		elif [ $rs_count -eq 0 ]
		then

		awk '{$1+=0}1' $parsingFolder/$folder_name/${file_name}${rs1}_tmp.cpnLong > $parsingFolder/$folder_name/${file_name}${rs1}.cpnLong
		
		fi
		
		rs1="${rs1}_rs"
		rs2="${rs2}_rs"
		rs_count=$(($rs_count+1))
		it_processed=$(($it_processed+$it_adjusted+1))

	done

fi		


echo "remove _tmp files"

rm $parsingFolder/$folder_name/${file_name}*tmp.cpnLong
rm $parsingFolder/$folder_name/${file_name}_cpn.it

echo "concat _rs files"

cat $parsingFolder/$folder_name/${file_name}*.cpnLong > $parsingFolder/$folder_name/${file_name}.all.cpnLong
mv $parsingFolder/$folder_name/${file_name}.all.cpnLong $parsingFolder/$folder_name/${file_name}.cpnLong
rm $parsingFolder/$folder_name/${file_name}_*.cpnLong
#### END

end_time=$(date -u +%s)
elapsed=$(($end_time-$start_time))
echo "Total time in sec: $elapsed"


