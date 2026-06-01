

TRAINING_DIR="/media/aris/Data/master2025dev/aris_master/training/03-17_Default_Constant_LR/checkpoint-45000"
OUTPUT_DIR="/media/aris/Data/master2025dev/datasets/synthetic/"
OUTPUT_DIR="/media/aris/Data/master2025dev/datasets/synthetic/03-17_Default_Constant_LR/1000_epochs"

# Define session name
MODEL_NAME=""

accelerate launch src/waste_diffuser/dataset_gen.py \
  --model_dir="${TRAINING_DIR}${MODEL_NAME}" \
  --output_dir="${OUTPUT_DIR}${MODEL_NAME}" \
  --batch_size=16 \
  --num_inference_steps=50 \
  --image_num=1500 
