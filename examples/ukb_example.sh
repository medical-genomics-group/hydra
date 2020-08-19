#!/bin/bash
# This is an example submission script for UKB biobank data.
# This is under an SLURM cluster
# by setting 10 nodes and 8 tasks per node we have a total of 80 tasks
# we have 4 threads per task
# we set a limit of 3 days
#SBATCH --job-name=annotmafLD
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time 3-00:00:00
#SBATCH --output groups72_mix4_unrelated_cpus4_tasks8_nodes10_mpisync10_height_4_unrelated_148covadjusted_w35520NA.log
#SBATCH --error groups72_mix4_unrelated_cpus4_tasks8_nodes10_mpisync10_height_4_unrelated_148covadjusted_w35520NA.err

mkdir -p ./ukb_height_72groups/

# to make full use of the 4 threads assigned by slurm
export OMP_NUM_THREADS=4

env | grep SLURM

# the random number seed
echo SEED = 214096108

start_time="$(date -u +%s)"
# here the path to hydra must be set
EXE=/hydra/src/hydra


# the following command is for a data set of 382466 individuals and 8430446 snps
# the option --mpibayes bayesMPI runs BayesRR, that is, BayesR with many groups of snps
# the option --pheno indicates the phenotype file
# the option --chain-length 100005 tells us we wish 10005 iterations
# we chose a --thin of 5, that is, we output the samples every 5 iterations
# we chose a --save of 1000 samples, that is, the whole state will be saved (and replaced) every 1000 samples
# with --mcmc-out-dir we chose the directory where output will be saved
# with --mcmc-out-name we chose the prefix for all the output files
# we set the random seed with --seed
# we shuffle markers with --shuf-mark 1, recommended
# we chose a --sync-rate of 10, i.e. we sinchronyse every 10 snps
# we chose the --groupIndexFile, one line per snp, indexes starting from 0
# we chose the --groupMixtureFile, groups separated by ; mixtures by ,
# we set the --bfile path, it must contain the .bed, .bim and .fam files
# we set the paht of --sparse-dir if any sparse index files
# we set a --threshold-fnz 0.060, i.e. above 6% MAF we use binary data, below sparse index
# sparse-sync, synchronisation using sparse messages, sometimes its faster
cmd="$EXE  --number-individuals 382466 --number-markers 8430446 --mpibayes bayesMPI --pheno /work/ukb_imp_v3_UKB_EST_oct19_pheno_w35520NA/ukb_imp_v3_UKB_EST_oct19_height_wNA_unrelated_148covadjusted_w35520NA.phen --chain-length 10005 --thin 5 --save 1000 --mcmc-out-dir /scratch/ukb_height_72groups --mcmc-out-name groups72_mix4_unrelated_cpus4_tasks8_nodes10_mpisync10_height_4_unrelated_148covadjusted_w35520NA --seed 214096108 --shuf-mark 1 --sync-rate 10 --groupIndexFile /work/ukb_imp_v3_UKB_EST_oct19_unrelated_annot12_maf3_ld2_bins.group --groupMixtureFile /work/ukb_imp_v3_UKB_EST_oct19_unrelated_annot12_maf3_ld2_bins_cpn_4.cva --bfile /work/ukb_imp_v3_UKB_EST_oct19/ukb_imp_v3_UKB_EST_oct19_unrelated --sparse-dir /work/sparse/ --sparse-basename ukb_imp_v3_UKB_EST_oct19_unrelated --threshold-fnz 0.060 --sparse-sync "

echo
echo $cmd
echo 

srun $cmd

end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "Total time in sec: $elapsed"

