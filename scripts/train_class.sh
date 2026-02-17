DATASET_DIR="/media/aris/Data/master2025dev/datasets"
TRAINING_DIR="/media/aris/Data/master2025dev/aris_master/training/"
# Define session name
SESSION_NAME="02-10_4-class_128"

accelerate launch ../src/waste_diffuser/training.py \
  --output_dir="${TRAINING_DIR}${SESSION_NAME}" \
  --train_data_dir="$DATASET_DIR/wood/4_main_categories_512x512/train/impregnated_wood/" \
  --config_path="config.json" \
  --resolution=128 --center_crop --random_flip \
  --train_batch_size=32 \
  --num_epochs=2000 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=bf16 \
  --checkpointing_steps=5000 \
  --comment="Conditional training with two datasets" \
  --resume_from_checkpoint="latest"