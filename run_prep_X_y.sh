#!/bin/bash
#SBATCH --job-name=pytorch_simeplDQ    # Job name
#SBATCH --mail-type=BEGIN,END,FAIL     # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yanweng@uci.edu    # Where to send mail
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --cpus-per-task=12         # CPU cores/threads
#SBATCH --mem=50gb                     # Job Memory
#SBATCH --partition=zhanglab.p           # use free-gpu partition
#SBATCH --output=serial_test_%j.log    # Standard output and error log



module load anaconda/2020.07
source activate snp_effect

DIR_source="/home/yanweng/snp_effect/data/general/pre_process"
DIR="/home/yanweng/snp_effect/data/for_model_v2/input"
X_GENERATOR=${DIR}/X2
Y_GENERATOR=${DIR}/y2

python /home/yanweng/snp_effect/script/deep_learning/model_v2_rsync_galaxy/prep_X_y.py ${DIR_source}/merged500_1kb_peak_labeled_hg19_self.csv ${DIR_source}/reverse_merged500_1kb_peak_labeled_hg19_self.csv ${DIR_source}/encode_negative.csv ${DIR_source}/merged500_1kb_peak_labled.bed ${X_GENERATOR} ${Y_GENERATOR}