#!/usr/bin/perl

use warnings;
use strict;
use File::Path qw(make_path remove_tree);

# Info on dataset to process
# --------------------------
my $DATADIR = "$ENV{HOME}/CADMOS/Matthew/BayesRRcmd/test/data/testdata_msp_constpop_Ne10K_M100K_N10K";
my $DATASET = "testdata_msp_constpop_Ne10K_M100K_N10K";
my $PHEN    = $DATASET;
my $S="1.0,0.1";

$DATADIR = "/scratch/orliac/memtest_M1000K_N50K";
$DATASET = "memtest_M1000K_N50K";
$PHEN    = $DATASET;
$S="0.1,0.01";

$DATADIR = "/scratch/orliac/testN500K";
$DATASET = "testN500K";
$PHEN    = $DATASET;
$S="0.1,0.01";


my $M       = 114560;      # Number of markers
$M          = 894417;
$M          = 1270420;
my $N       = 500000;       # Number of individuals
my $CL      = 20000;       # Number of iterations (chain length)
my $SM      = 1;           # Marker shuffling switch

my $MEMGB   = 180;         # Helvetios

#my $EXE     = "/home/orliac/DCSR/CTGG/BayesRRcmd/src/mpi_gibbs"; # Binary to run
my $EXE     = "/home/orliac/DCSR/CTGG/BayesRRcmd/src/mpi_gibbs_devel";
die unless -d $DATADIR;
die unless (-f $EXE && -e $EXE);

# Benchmark plan & processing setup
# ---------------------------------
my $nickname = "large_10K_it_cinq"; #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHANGE THIS ONE !!!!!!!

my $DIR    = "/scratch/orliac/test_sparse/benchmarks/$nickname";
unless (-d $DIR) { make_path($DIR) or die "mkdir($DIR): $!"; }
my @NNODES = qw(32 16);
my @NTPN   = qw(36);
my @SYNCR  = qw(5);
my @SEEDS  = qw(5432);

my $PARTITION = 'parallel';
#$PARTITION = 'debug';


my $submit = "$DIR/submit_all_sbatch_$nickname.sh";
open S, ">$submit" or die $!;

foreach my $nnodes (@NNODES) {

    foreach my $ntpn (@NTPN) {

        my $ntasks = $nnodes * $ntpn;
        printf("Total number of tasks: %4d ( = %3d x %3d)\n", $ntasks, $ntpn, $nnodes);

        # Assuming 1 CPU per task
        my $cpu_per_task = 1;
        my $mem_per_node = $MEMGB;

        foreach my $syncr (@SYNCR) {

            foreach my $SEED (@SEEDS) {

                #printf("nodes: $nnodes, tasks per node: $ntpn, syncr: $syncr\n");
                my $basename = sprintf("${nickname}__nodes_%02d__tpn_%02d__tasks_%02d__cl_${CL}__syncr_%03d__seed_%02d", $nnodes, $ntpn, $ntasks, $syncr, $SEED);
                
                # Delete output file if already existing
                my $csv = "$DIR/$basename".'.csv';
                if (-f $csv) {
                    unlink $csv;
                    print "INFO: deleted file $csv\n";
                }

                my $bet = "$DIR/$basename".'.bet';
                if (-f $bet) {
                    unlink $bet;
                    print "INFO: deleted file $bet\n";
                }
                
                open F, ">$DIR/$basename.sh" or die $!;

                print F "#!/bin/bash\n\n";
                print F "#SBATCH --nodes $nnodes\n";
                print F "#SBATCH --exclusive\n";
                print F "#SBATCH --mem ${mem_per_node}G\n";
                print F "#SBATCh --ntasks $ntasks\n";
                print F "#SBATCH --ntasks-per-node $ntpn\n";
                print F "#SBATCH --cpus-per-task 1\n";
                print F "#SBATCH --time 3-00:00:00\n";
                #print F "#SBATCH --time 06:00:00\n";
                print F "#SBATCH --partition $PARTITION\n";
                #print F "#SBATCH --constraint=E5v4\n";
                print F "#SBATCH --output ${basename}__jobid\%J.out\n";
                print F "#SBATCH --error  ${basename}__jobid\%J.err\n";
                print F "\n";
                print F "module load intel intel-mpi eigen boost\n\n";
                print F "env | grep SLURM\n\n";
                print F "\n";
                print F "start_time=\"\$(date -u +\%s)\"\n";
                print F "\n";
                #print F "srun $EXE --bfile $DATADIR/$DATASET --pheno $DATADIR/$DATASET.phen --chain-length $CL --seed $SEED --shuf-mark $SM --mpi-sync-rate $syncr --number-markers $M --mcmc-samples ${basename}.csv\n";
                print F "srun $EXE --mpibayes bayesMPI --bfile $DATADIR/$DATASET --pheno $DATADIR/${PHEN}.phen2 --chain-length $CL --burn-in 0 --thin 1 --mcmc-samples $csv --mcmc-betas $bet --seed $SEED --shuf-mark $SM --mpi-sync-rate $syncr --number-markers $M --S $S\n";
                print F "\n";
                print F "end_time=\"\$(date -u +\%s)\"\n";
                print F "elapsed=\"\$((\$end_time-\$start_time))\"\n";
                print F "echo \"Total time in sec: \$elapsed\"\n";
                close F;
                print S "sbatch $basename.sh\n";
            }
        }
    }
}

close S;

print "\nWrote $submit. To submit: sh $submit\n\n";
