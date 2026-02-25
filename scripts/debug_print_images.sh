DATASET_DIR="/media/aris/Data/master2025dev/datasets"
TRAINING_DIR="/media/aris/Data/master2025dev/aris_master/training/"
# Define session name
SESSION_NAME="02-10_4-class_128"

accelerate launch ../src/waste_diffuser/training.py \
  --output_dir="${TRAINING_DIR}${SESSION_NAME}" \
  --train_data_dir="$DATASET_DIR/wood/4_main_categories_512x512/train/impregnated_wood/" \
  --config_path="config_template.json" \
  --resolution=128 --center_crop --random_flip \
  --train_batch_size=32 \
  --num_epochs=2000 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1.41e-4 \
  --lr_warmup_steps=500 \
  --lr_scheduler="cosine_with_restarts" \
  --mixed_precision=bf16 \
  --checkpointing_steps=5000 \
  --save_images_epochs=20 \
  --comment="-" \
  --resume_from_checkpoint="latest"