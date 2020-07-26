#!/bin/bash
#SBATCH -o output.generate_truth.out # STDOUT
#SBATCH --job-name=truth_table   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G        # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=12:00:00           # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=jdezoort@princeton.edu

module purge
module load anaconda3
conda activate gnnenv

cd /home/jdezoort/General_Repo_Dev/princeton_gnn_tracking/analyses/truth
python generate_truth_LP.py /tigress/jdezoort/train_1
