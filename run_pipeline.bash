#!/bin/tcsh -e
#SBATCH --job-name=deblur_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=parikhnf@bc.edu
#SBATCH --partition=full_nodes64

#SBATCH --output=pipeline_%j.txt

module load cuda11.2
module load anaconda 

conda activate med_deblur

hostname

python main.py --filepaths '../file_list.txt' --threshold 100 --output_root_dir '../synthetic_dataset/0025'