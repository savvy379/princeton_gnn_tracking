#!/bin/bash
#SBATCH -o /tigress/jdezoort/IN_output/job_reports/output.prep_LP_1_EC.out  # STDOUT
#SBATCH --job-name=LP_prep_1_EC       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=128G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=jdezoort@princeton.edu

module purge
module load anaconda3
conda activate gnnenv

cd /home/jdezoort/General_Repo_Dev/princeton_gnn_tracking/
python prepare_LP_IN.py configs/IN/prepare_LP.yaml
