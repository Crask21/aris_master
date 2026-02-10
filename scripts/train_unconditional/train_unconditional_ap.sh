DATASET_DIR="/media/aris/Data/master2025dev/datasets"
TRAINING_DIR="/media/aris/Data/master2025dev/aris_master/training/"

# Define session name
SESSION_NAME="03-02_ddim_ema_impregnated-wood_128_2x-self-attention-DELETE_LATER"

accelerate launch train_unconditional_ap.py \
  --output_dir="${TRAINING_DIR}${SESSION_NAME}" \
  --train_data_dir="$DATASET_DIR/wood/4_main_categories_512x512/train/impregnated_wood/" \
  --resolution=128 --center_crop --random_flip \
  --train_batch_size=16 \
  --num_epochs=1000 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --checkpointing_steps=2000 \
  --comment="Let's try 1000 epochs. Worked for that guy." \
  #--resume_from_checkpoint="latest"

  python3 inference.py \
    --model_dir="${TRAINING_DIR}${SESSION_NAME}" \
    --batch_size=16 \
    --num_inference_steps=50 \
    --image_num=1



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