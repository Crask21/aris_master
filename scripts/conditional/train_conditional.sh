# accelerate launch train_unconditional.py \
#   --train_data_dir="/media/aris/Data/master2025dev/datasets/wood/4_main_categories_512x512/train/impregnated_wood/images/" \
#   --resolution=256 --center_crop --random_flip \
#   --output_dir="/media/aris/Data/master2025dev/aris_master/training/ddim-ema-impregnated-wood-256" \
#   --train_batch_size=32 \
#   --num_epochs=100 \
#   --gradient_accumulation_steps=1 \
#   --use_ema \
#   --learning_rate=1e-4 \
#   --lr_warmup_steps=500 \
#   --mixed_precision=no 

# Perhaps run more sequentially:

DATASET_DIR="/media/aris/Data/master2025dev/datasets"
TRAINING_DIR="/media/aris/Data/master2025dev/aris_master/training/"

# Define session name
SESSION_NAME="07-02_2-class-conditioning-wood-plastic_128"

accelerate launch train_conditional.py \
  --output_dir="${TRAINING_DIR}${SESSION_NAME}" \
  --train_data_dir="$DATASET_DIR/wood/4_main_categories_512x512/train/impregnated_wood/" \
  --train_data_dir2="/media/aris/Data/master2025dev/datasets/plastic/all_plastic_categories/hard_plastic"\
  --resolution=128 --center_crop --random_flip \
  --train_batch_size=32 \
  --num_epochs=1000 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --checkpointing_steps=5000 \
  --comment="Conditional training with two datasets" \
  --resume_from_checkpoint="latest"

  # python3 inference.py \
  #   --model_dir="${TRAINING_DIR}${SESSION_NAME}" \
  #   --batch_size=16 \
  #   --num_inference_steps=50 \
  #   --image_num=1 \
  #   --vae



# accelerate launch train_unconditional.py \
#   --train_data_dir="/media/aris/Data/master2025dev/datasets/drink_can_dataset_512x512/" \
#   --resolution=64 --center_crop --random_flip \
#   --output_dir="ddim-ema-drink-can-64" \
#   --train_batch_size=16 \
#   --num_epochs=100 \
#   --gradient_accumulation_steps=1 \
#   --use_ema \
#   --learning_rate=1e-4 \
#   --lr_warmup_steps=500 \
#   --mixed_precision=no 