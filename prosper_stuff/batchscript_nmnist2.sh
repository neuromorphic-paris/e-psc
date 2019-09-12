#!/bin/bash
#SBATCH --job-name="nmnist"
#SBATCH --output=nmnist-%j.out
#SBATCH --error=nmnist-%j.err
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100M
# mail alert at start, end and abortion of execution
#SBATCH --mail-user=georgios.exarchakis@uni-oldenburg.de
#SBATCH --mail-type=ALL
#
#SBATCH --partition=gold
#SBATCH --time=48:00:00
###SBATCH --gres=gpu:0
###SBATCH --nodes=1
#SBATCH --nodelist=gold0[1-3],gold[05-11]
###SBATCH --export=ALL
#
### Keep in mind to load your desired modules here. Otherwise they won't be available in your Job. Read more about modules at Environment modules
#module load openmpi/gcc/3.0.0
#module load impi/4.1.3.048/32/gcc 
#module load impi/4.1.3.048/64/gcc 
#module load impi/4.1.3.048/64/intel
#module load impi/5.1.3.210/64/gcc
#module load impi/5.1.3.210/64/intel 
#module load impi/5.1.3.210/64/gcc
module load impi
module load python/conda-root
conda init zsh
conda init bash
#module load python/3.6.4-ompi
conda activate psc2
#
##Here should be the command you want to execute. For example :
echo `which python`
mpirun  ${HOME}/.conda/envs/psc2/bin/python main_nmnist2.py
#mpiexec python main_nmnist.py


