

TRAINING_DIR="/media/aris/Data/master2025dev/aris_master/training/"
OUTPUT_DIR="/media/aris/Data/master2025dev/datasets/synthetic/"

# Define session name
MODEL_NAME="04-02_ddim_ema_vae_impregnated-wood_128"

accelerate launch dataset_gen.py \
  --model_dir="${TRAINING_DIR}${MODEL_NAME}" \
  --output_dir="${OUTPUT_DIR}${MODEL_NAME}" \
  --batch_size=16 \
  --num_inference_steps=50 \
  --image_num=1000