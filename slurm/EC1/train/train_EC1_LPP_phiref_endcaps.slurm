#!/bin/bash                                                                                                                          
#SBATCH -o /home/sthais/trkx/heptrkx-gnn-tracking/job_outputs/output.EC1_LPP_600epoch_4iter_phiref_endcaps.out # STDOUT     
#SBATCH --job-name=LPP_Train      # create a short name for your job                                                                  
#SBATCH --nodes=1                # node count                                                                                        
#SBATCH --ntasks=1               # total number of tasks across all nodes                                                            
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=160G        # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=50:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=sthais@princeton.edu

module purge
module load anaconda3
conda activate gnn3

cd /home/sthais/trkx/heptrkx-gnn-tracking
python train_EC1.py configs/train_ec1_lpp_phiref_endcaps.yaml
#python plot_IN.py configs/train_IN.yaml
