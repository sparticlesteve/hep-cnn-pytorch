export OMP_NUM_THREADS=64
export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"
export KMP_BLOCKTIME=1
module load pytorch-mpi/v0.4.0
