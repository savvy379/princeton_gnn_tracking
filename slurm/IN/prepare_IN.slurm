#!/bin/bash
#SBATCH -o /tigress/jdezoort/IN_output/job_reports/output.prep_LP.out  # STDOUT
#SBATCH --job-name=LP_prep       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=06:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=jdezoort@princeton.edu

module purge
module load anaconda3
conda activate gnnenv

cd /home/jdezoort/General_Repo_Dev/princeton_gnn_tracking/
python prepare_LP.py configs/IN/prepare_LP.yaml
