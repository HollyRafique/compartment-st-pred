#!/bin/bash

#$ -l tmem=50G
#$ -pe smp 8
#$ -R y
#$ -l h_rt=120:00:00
#$ -j y
#$ -N train_cd45
#$ -l h="!(*arbu*|*gonzo*|*webb*|*sweetums*|*korman*|*saunders*|*abner*|*lucky*|*newman*)"
set -eo pipefail

# Capture the hostname in a variable
host=$(hostname)
echo "host: $host"

date

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export PATH=/home/hrafique/.local/bin:$PATH
source /share/apps/source_files/python/python-3.11.9.source

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

input_path="/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/features/original"
truth_path="/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/truthlabels"
output_path="/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/output"
label_file="svgs_cd45_raw1.csv"

segment="CD45+"
foundation_model="combined"
feature_file="${foundation_model}-${segment}_features.csv"
config_file="/home/hrafique/compartment-st-pred/config/config_genetraining.yaml"

python3 /home/hrafique/compartment-st-pred/src/train_gene_expression.py \
    -ip "$input_path" -tp "$truth_path" -op "$output_path" \
    -tf "$label_file" -ff "$feature_file" -cp "$config_file" \
    -fm "$foundation_model" -s "$segment" \
    --n_workers "${NSLOTS:-1}"


date
