#!/usr/bin/env bash

TRAINING_DIR="/media/aris/Data/master2025dev/aris_master/training/04-15_masked_diffusion/1000_impregnated_wood_only"
TRAINING_DIR="/media/aris/Data/master2025dev/aris_master/training/04-15_masked_diffusion/100"
OUTPUT_DIR="/media/aris/Data/master2025dev/datasets/aris_4_class/1000_real/masked_diffusion/100/"
# OUTPUT_DIR="/media/aris/Data/master2025dev/datasets/synthetic/03-17_Default_Constant_LR/10000_epochs"

# Define session name
MODEL_NAME=""

accelerate launch src/waste_diffuser/dataset_gen.py \
  --model_dir="${TRAINING_DIR}${MODEL_NAME}" \
  --output_dir="${OUTPUT_DIR}${MODEL_NAME}" \
  --batch_size=16 \
  --num_inference_steps=50 \
  --image_num=160 \
  --method="masked" \
  --dataset_path="/media/aris/Data/master2025dev/datasets/aris_4_class/1000_real/real/train" \
  --classes 'impregnated_wood' 'normal_wood' 'hard_plastic' 'soft_plastic' 
