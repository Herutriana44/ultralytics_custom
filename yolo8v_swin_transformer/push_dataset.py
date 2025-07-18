import os
from huggingface_hub import HfApi, login
from pathlib import Path
import shutil
import tempfile

HUGGINGFACE_API = "hf_kxxxxxxxxxxxxxxxx" #@param

# Login to Hugging Face (akan meminta token)
print("Please login to Hugging Face...")
login(token=HUGGINGFACE_API)

api = HfApi()
repo_id = "herutriana44/medical-detection-dataset"
dataset_path = "../dataset_medis"

# Create the dataset repo if it doesn't exist
try:
    api.create_repo(
        repo_id=repo_id, 
        repo_type="dataset",
        exist_ok=True
    )
    print(f"Dataset repository created/exists: {repo_id}")
except Exception as e:
    print(f"Error creating dataset repo: {e}")
    exit(1)

# Create temporary directory for dataset
with tempfile.TemporaryDirectory() as temp_dir:
    temp_dataset_path = os.path.join(temp_dir, "dataset_medis")
    os.makedirs(temp_dataset_path)
    
    # Copy images and labels
    images_train_path = os.path.join(dataset_path, "images", "train")
    images_val_path = os.path.join(dataset_path, "images", "val")
    labels_train_path = os.path.join(dataset_path, "labels", "train")
    labels_val_path = os.path.join(dataset_path, "labels", "val")
    
    if os.path.exists(images_train_path):
        shutil.copytree(images_train_path, os.path.join(temp_dataset_path, "images", "train"))
        print("Copied train images")
    if os.path.exists(images_val_path):
        shutil.copytree(images_val_path, os.path.join(temp_dataset_path, "images", "val"))
        print("Copied val images")
    if os.path.exists(labels_train_path):
        shutil.copytree(labels_train_path, os.path.join(temp_dataset_path, "labels", "train"))
        print("Copied train labels")
    if os.path.exists(labels_val_path):
        shutil.copytree(labels_val_path, os.path.join(temp_dataset_path, "labels", "val"))
        print("Copied val labels")
    
    # Copy YAML file
    yaml_path = os.path.join(dataset_path, "data_medis.yaml")
    if os.path.exists(yaml_path):
        shutil.copy2(yaml_path, os.path.join(temp_dataset_path, "data_medis.yaml"))
        print("Copied YAML file")
    
    # Upload dataset folder
    try:
        api.upload_folder(
            folder_path=temp_dataset_path,
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"Dataset uploaded successfully to: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        print("Trying with upload_large_folder...")
        try:
            api.upload_large_folder(
                folder_path=temp_dataset_path,
          repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"Dataset uploaded successfully to: https://huggingface.co/datasets/{repo_id}")
        except Exception as e2:
            print(f"Error with upload_large_folder: {e2}")