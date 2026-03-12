#!/bin/bash
#SBATCH --account=rrg-mbowling-ad
#SBATCH --cpus-per-task=1 
#SBATCH --gpus-per-node=1 
#SBATCH --mem=16G 
#SBATCH --time=0-2:59
#SBATCH --array=1-30
#SBATCH --output=model_free/vanilla_count_model_free_%j.out



echo "Starting task $SLURM_ARRAY_TASK_ID"
 
module load python/3.10 StdEnv/2023 gcc opencv/4.8.1 swig mujoco/3.2.2

cd $SLURM_TMPDIR
mkdir $SLURM_JOB_ID
cd $SLURM_JOB_ID

export ALL_PROXY=socks5h://localhost:8888


git clone https://github.com/artemis79/curious-replay-counts.git

#Install requirements
python -m venv .venv
source .venv/bin/activate

cd curious-replay-counts
pip install -r requirements.txt






