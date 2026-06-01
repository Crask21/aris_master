from src.waste_diffuser.json_mask_to_tensor import json_mask_to_tensor
import os
import torch
from pathlib import Path


def main():
    dataset_list = ["25_real", "50_real", "100_real", "500_real", "1000_real"]
    # dataset_list = ["10_real"]
    
    for dataset in dataset_list:
        print(f"Processing dataset: {dataset}")
        json_path = f"/media/aris/Data/master2025dev/datasets/aris_4_class/{dataset}/real/"
        
        #for every .json file in the directory and all subdirectories, convert to tensor and save as .pt file in the same location
        for root, dirs, files in os.walk(json_path):
            for file in files:
                if file.endswith(".json"):
                    json_file = os.path.join(root, file)
                    print(f"Processing file: {json_file}")
                    try:
                        tensor = json_mask_to_tensor(json_file)
                        pt_file = json_file.replace("annots", "masks").replace("annot", "mask").replace(".json", ".pt")
                        os.makedirs(os.path.dirname(pt_file), exist_ok=True)
                        torch.save(tensor, pt_file)
                        print(f"Saved tensor to: {pt_file}")
                    except Exception as e:
                        print(f"Error processing {json_file}: {e}")
        



if __name__ == "__main__":
    main()