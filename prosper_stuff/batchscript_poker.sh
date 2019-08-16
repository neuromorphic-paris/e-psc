#!/bin/bash
#SBATCH --job-name="poker"
#SBATCH --output=poker-%j.out
#SBATCH --error=poker-%j.err
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000M
# mail alert at start, end and abortion of execution
#SBATCH --mail-user=georgios.exarchakis@uni-oldenburg.de
#SBATCH --mail-type=ALL
#
#SBATCH --partition=gold
#SBATCH --time=48:00:00
###SBATCH --gres=gpu:0
###SBATCH --nodes=1
###SBATCH --nodelist=gold0[1-9]
###SBATCH --export=ALL
#
### Keep in mind to load your desired modules here. Otherwise they won't be available in your Job. Read more about modules at Environment modules
module load python/conda-root
#conda init zsh
#conda init bash
#module load python/3.6.4-ompi
source activate psc
module load impi
#
##Here should be the command you want to execute. For example :
mpirun -iface bond0 python main_h5.py

