export OMP_NUM_THREADS=32
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1

# Trying my master build
export MPICH_MAX_THREAD_SAFETY=multiple
conda activate /global/cscratch1/sd/sfarrell/conda/pytorch-mpi/b0ad810
