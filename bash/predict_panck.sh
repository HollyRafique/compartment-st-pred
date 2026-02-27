#!/bin/bash

#$ -l tmem=50G
# -l gpu=true
# -pe gpu 1
#$ -R y
#$ -l h_rt=120:00:00
#$ -j y
#$ -N 'st_OpHc2'
#$ -l h="!(*arbu*)"

hostname
# Capture the hostname in a variable
host=$(hostname)
echo "host: $host"

date

#export LD_LIBRARY_PATH=/share/apps/onetbb-2021.1.1/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/share/apps/openslide-3.4.1/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export PATH=/home/hrafique/.local/bin:$PATH
source /share/apps/source_files/python/python-3.11.9.source

input_path="/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/features/original"
truth_path="/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/truthlabels"
output_path="/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/output/SVMonly"
label_file="hvgs_panck_raw2.csv"

segment="PanCK+"
foundation_model="combined"
feature_file="${foundation_model}-${segment}_features.csv"
config_file="/home/hrafique/multimodal/config/config_rbf.yaml"

python3 /home/hrafique/multimodal/src/predict_sge.py -ip $input_path -tp $truth_path -op $output_path \
                                                     -tf $label_file -ff $feature_file -cp $config_file \
                                                     -fm $foundation_model  -s $segment


date
