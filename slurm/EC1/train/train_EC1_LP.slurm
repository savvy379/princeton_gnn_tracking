#!/bin/bash                                                                                                                          
#SBATCH -o /home/sthais/trkx/heptrkx-gnn-tracking/job_outputs/output.EC1_LP_600.out # STDOUT     
#SBATCH --job-name=LP_Train      # create a short name for your job                                                                  
#SBATCH --nodes=1                # node count                                                                                        
#SBATCH --ntasks=1               # total number of tasks across all nodes                                                            
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)                                                   
#SBATCH --mem-per-cpu=190G        # memory per cpu-core (4G per cpu-core is default)                                                  
#SBATCH --time=30:00:00          # total run time limit (HH:MM:SS)                                                                   
#SBATCH --mail-type=all          # send email on job start, end and fail                                                             
#SBATCH --mail-user=jdezoort@princeton.edu                                                                                                     

module purge
module load anaconda3
conda activate gnn3

cd /home/sthais/trkx/heptrkx-gnn-tracking
python train_EC1.py configs/train_ec1.yaml
#python plot_IN.py configs/train_IN.yaml
