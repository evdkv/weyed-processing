#!/bin/bash

#SBATCH --job-name=gazeDS_proc
#$$$$$$$$$$$$$$$$$
# the account is either your netid if you are faculty, or your
# faculty sponsor's netid if you are not.
#$$$$$$$$$$$$$$$$$
#SBATCH --account=arobbins

#$$$$$$$$$$$$$$$$$
# You can change the start time to something like "saturday" or
# "now+4hours" or "tomorrow". You can also use a full ISO date
# written like YYYY-MM-DDTHH:MM:SS  Note the T between the
# date and the time.
#$$$$$$$$$$$$$$$$$
#SBATCH --begin=now
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tolya.evdokimov@richmond.edu

#$$$$$$$$$$$$$$$$
# Most of the time, you are only running one program at a time,
# so --ntasks is almost always 1. --mem (memory) is up to you,
# as is --cpus-per-task. --cpus-per-task is really the number
# of cores.
#$$$$$$$$$$$$$$$$
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=12

#$$$$$$$$$$$$$$$$$$$
# The community partitions are:
#  basic, medium, large, ML, and sci.
# Consult your faculty sponsor. The basic partition is the default,
# and will be used if you omit this line.
#$$$$$$$$$$$$$$$$$$$

#$$$$$$$$$$$$$$$$$$$$$
# This figure is the execution time in hours:minutes:seconds.
# The current limit is 8760:00:00, which is one year.
#$$$$$$$$$$$$$$$$$$$$
#SBATCH --time=60:00:00

#$$$$$$$$$$$$$$$$$$$$
# -o is the stdout of your job, what you would see on the screen.
# -e is the stderr of your job. Errors, diagnostics, debug, and other.
# The default values will guarantee uniqueness. Note that if you
# do not provide a -e file, all the output will be combined into
# one file.
#$$$$$$$$$$$$$$$$$$$$
#SBATCH -o slurm-%j.out
# #SBATCH -e slurm-%j.err

echo "I ran on: $SLURM_NODELIST"
echo "Starting at `date`"

###
# Environment setup. You can change these to whatever you like.
# Note that the NAME is what SLURM will call your job. 
###

export ME=`/usr/bin/whoami`

#$$$$$$$$$$$$$$$$$$$$$$$$$
# You should change these variables to reflect the reality of 
# what you are doing. You are not required to use these at all;
# you can provide the exact path name for all files (although
# many scientific systems use $SCRATCH.
#$$$$$$$$$$$$$$$$$$$$$$$$$
export DATADIR=$HOME
export SCRATCH=/localscratch/$ME
export BIGSCRATCH=/scratch/$ME

########################################################################
# SLURM starts with *nothing*, so you need to define your environment
# variables, and load your modules (if you use modules).
########################################################################

#$$$$$$$$$$$$$$$
# This is the location of the module files on Spydur
#$$$$$$$$$$$$$$$
export MODULEPATH="$MODULEPATH:/usr/local/sw/modulefiles"


#$$$$$$$$$$$$$$$$$$$
# Always use mkdir -p ... it will not fail if the directory already
# exists.
#$$$$$$$$$$$$$$$$$$$
mkdir -p $SCRATCH
mkdir -p $BIGSCRATCH

########################################################################
########################################################################
# Run your job by adding commands below for your program. 
# You may need to set some additional ENV variables, or load 
# additional modules before you add the executable commands.
# 
# The echos are just placeholders.
########################################################################
########################################################################

# echo "Setting up conda..."
# conda init bash
# source ~/.bashrc
# conda activate cs395

# echo "Setting up cuda..."
# export CUDA_HOME="/usr/local/cuda-11.4"
# export PATH="$CUDA_HOME:$PATH"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# /opt/nvidia/hpc_sdk/Linux_x86_64/21.5/cuda/11.3/bin/nvcc --version

source eye_env/bin/activate
echo "Running process.py"
python process.py
echo "Running prepare.py"
python prepare.py
echo "Running serialize_to_tfrecord.py"
python serialize_to_tfrecord.py
    
#############################################
# Now copy your results back from the node (if there are any)
#############################################
if [ ! -z "$(ls -A $SCRATCH)" ]; then
    cp -r $SCRATCH/* $BIGSCRATCH
fi 

########################################################################
# Be kind and clean the SCRATCH area.
########################################################################
rm -rf $SCRATCH

echo "Finished at `date`"

