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

accelerate launch train_unconditional.py \
  --train_data_dir="/media/aris/Data/master2025dev/datasets/drink_can_dataset_512x512/" \
  --resolution=128 --center_crop --random_flip \
  --output_dir="/media/aris/Data/master2025dev/aris_master/training/ddim-ema-drink-can-128" \
  --train_batch_size=64 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no 

# Perhaps run more sequentially: