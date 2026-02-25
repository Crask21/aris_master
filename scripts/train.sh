DATASET_DIR="/media/aris/Data/master2025dev/datasets"
TRAINING_DIR="/media/aris/Data/master2025dev/aris_master/training/"

# accelerate launch ../src/waste_diffuser/training.py \
#   # --config_path="diffusion_1_layer.json" 
#   --config_path="config_template.json" 
python ../src/waste_diffuser/training.py \
  --config_path="diffusion_1_layer.json"


# ------------------------ USE THE FOLLOWING FOR QUEUE ----------------------- #
# accelerate launch ../src/waste_diffuser/training.py \
#   --config_path=$CONFIG 