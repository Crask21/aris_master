import shutil
import argparse
from pathlib import Path
from datetime import datetime

from classification.utils.data import DataClass
from classification.utils.config import Config

def parse_args():
    parser = argparse.ArgumentParser("Config file creation", description="This script is used to create a config file.")
    parser.add_argument("-f", "--data_file", type=str, help="path to the yaml file.", required=True)
    parser.add_argument("-t", "--tag", type=str, help="tag for the config file.", required=True)
    parser.add_argument("--class-separation", type=str, default=None, help="path to class separation yaml file (optional).")

    return parser.parse_args()

if __name__ == '__main__':
    """
    This script creates a configuration file based on the provided data file.
    It reads the data file, extracts class weights, and saves a configuration file with the class names and their respective weights.
    It saves the config file and copies the data file to a specific directory structure as follows:
    classification/
        data/
            configs/
                wood/
                    2025-08-15T16_13/
                        wood_2025-08-15T16_13_config.yaml
                        wood_2025-08-15T16_13_dataset.yaml
                plastic/
                mineral_wool/
    """

    #  aris create_config_from_data_yaml.py -t wood -f /home/simon/Desktop/classification/2026-01-Wood/Classification_report/wood_2026-01-30T16_35_dataset.yaml
    #args = ["-t", "wood", "-f", "/home/simon/Desktop/classification/2026-01-Wood/Classification_report/wood_2026-01-30T16_35_dataset.yaml"]
    args = parse_args()
    DATA_FILEPATH = Path(args.data_file).absolute()
    TAG = args.tag

    datetime_str = datetime.now().strftime("%Y-%m-%dT%H_%M")
    config_and_data_dir = Path(__file__).parent.parent.parent / 'data' / 'configs' / TAG / datetime_str
    config_and_data_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{TAG}_{datetime_str}"
    config_file_path = config_and_data_dir / f"{file_name}_config.yaml"
    dataset_file_path = config_and_data_dir / f"{file_name}_dataset.yaml"

    print(f'Will save config file at  : {config_file_path}')
    print(f'Will copy dataset file to : {dataset_file_path}')
    
    data = DataClass.from_file(DATA_FILEPATH)

    config_file = Config()
    class_weights_dict = data.class_weights()

    config_file.model.data_file = str(DATA_FILEPATH)
    config_file.model.cls_names = list(class_weights_dict.keys())
    config_file.model.class_imbalance = list(class_weights_dict.values())
    config_file.logging.tag = TAG
    config_file.evaluation.output_dir = str(config_and_data_dir)

    config_file.save_config(config_file_path)
    shutil.copy(DATA_FILEPATH, dataset_file_path)
    
    print(f"Configuration file saved at {config_file_path}")
    print(f"Dataset file copied to {dataset_file_path}")
    print(config_file)

    print("Configuration and dataset files have been created successfully.")

