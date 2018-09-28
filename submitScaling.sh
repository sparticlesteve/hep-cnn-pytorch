#!/bin/bash

sbatch -J hep_cnn -d singleton -N 1 ./batchScript.sh
sbatch -J hep_cnn -d singleton -N 2 ./batchScript.sh
sbatch -J hep_cnn -d singleton -N 4 ./batchScript.sh
sbatch -J hep_cnn -d singleton -N 8 ./batchScript.sh
sbatch -J hep_cnn -d singleton -N 16 ./batchScript.sh
sbatch -J hep_cnn -d singleton -N 32 ./batchScript.sh
sbatch -J hep_cnn -d singleton -N 64 ./batchScript.sh

#sbatch -J hep_cnn -d singleton -q regular -N 128 ./batchScript.sh
#sbatch -J hep_cnn -d singleton -q regular -N 256 ./batchScript.sh
