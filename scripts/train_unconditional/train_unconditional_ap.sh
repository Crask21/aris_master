accelerate launch train_unconditional_ap.py \
  --train_data_dir="/media/aris/Data/master2025dev/datasets/wood/4_main_categories_512x512/train/normal_wood/" \
  --resolution=128 --center_crop --random_flip \
  --output_dir="/media/aris/Data/master2025dev/aris_master/training/ddim-ema-normal-wood-128-2x_self_attention-lr=1e-3" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-3 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --checkpointing_steps=2000 
  #--resume_from_checkpoint="latest"



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