#!/bin/bash
#SBATCH --job-name=cluster4_7_yg_pytorch      # Job name
#SBATCH --mail-type=BEGIN,END,FAIL     # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yanweng@uci.edu    # Where to send mail
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --cpus-per-task=12         # CPU cores/threads
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=100gb                     # Job Memory
#SBATCH --partition=zhanglab.p           # use free-gpu partition
#SBATCH --output=serial_test_%j.log    # Standard output and error log


#module load cuda/11.1
#module load conda/2020.11
#export PATH=/pkg/anaconda3/2020.11/bin:/usr/local/cuda-11.1/bin:$PATH
echo $PATH
source activate snp_effect_py3.7

python /home/yanweng/snp_effect/script/deep_learning/model_v2_rsync_galaxy/main.py -e 'train' -s 'main' -c '/home/yanweng/snp_effect/script/deep_learning/model_v2_rsync_galaxy/config/config_complexDQ_galaxy_cluster4_TRANS.json'

python /home/yanweng/snp_effect/script/deep_learning/model_v2_rsync_galaxy/main.py -e 'test' -s 'main' -c '/home/yanweng/snp_effect/script/deep_learning/model_v2_rsync_galaxy/config/config_complexDQ_galaxy_cluster4_TRANS.json'

python /home/yanweng/snp_effect/script/deep_learning/model_v2_rsync_galaxy/main.py -e 'train' -s 'main' -c '/home/yanweng/snp_effect/script/deep_learning/model_v2_rsync_galaxy/config/config_complexDQ_galaxy_cluster5_TRANS.json'

python /home/yanweng/snp_effect/script/deep_learning/model_v2_rsync_galaxy/main.py -e 'test' -s 'main' -c '/home/yanweng/snp_effect/script/deep_learning/model_v2_rsync_galaxy/config/config_complexDQ_galaxy_cluster5_TRANS.json'

python /home/yanweng/snp_effect/script/deep_learning/model_v2_rsync_galaxy/main.py -e 'train' -s 'main' -c '/home/yanweng/snp_effect/script/deep_learning/model_v2_rsync_galaxy/config/config_complexDQ_galaxy_cluster6_TRANS.json'

python /home/yanweng/snp_effect/script/deep_learning/model_v2_rsync_galaxy/main.py -e 'test' -s 'main' -c '/home/yanweng/snp_effect/script/deep_learning/model_v2_rsync_galaxy/config/config_complexDQ_galaxy_cluster6_TRANS.json'

python /home/yanweng/snp_effect/script/deep_learning/model_v2_rsync_galaxy/main.py -e 'train' -s 'main' -c '/home/yanweng/snp_effect/script/deep_learning/model_v2_rsync_galaxy/config/config_complexDQ_galaxy_cluster7_TRANS.json'

python /home/yanweng/snp_effect/script/deep_learning/model_v2_rsync_galaxy/main.py -e 'test' -s 'main' -c '/home/yanweng/snp_effect/script/deep_learning/model_v2_rsync_galaxy/config/config_complexDQ_galaxy_cluster7_TRANS.json'