#!/bin/bash

#PBS -l procs=1,gpus=1
#PBS -l walltime=01:00:00
#PBS -q p100_normal_q
#PBS -A mfulearn
#PBS -W group_list=newriver
#PBS -M yangzeng@vt.edu
#PBS -m bea
#PBS -N ABC_AE_NS
#PBS -j oe
#PBS -o Noise000.log

cd $PBS_O_WORKDIR

module purge
module load Anaconda/5.1.0
module load cuda/9.0.176
module load cudnn/7.1

source ../../src/init
ABC-AE-NS-Nonlinear-4.py setting sensitive_data.mat observation_dynamic.mat
