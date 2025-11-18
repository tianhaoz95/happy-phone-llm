from huggingface_hub import snapshot_download
import os

model_name = "Qwen/Qwen3-0.6B"
local_dir = "Qwen3-0.6B" # Relative to the model/ directory

# Ensure the target directory exists
os.makedirs(local_dir, exist_ok=True)

print(f"Downloading {model_name} to {local_dir}...")
snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
print("Download complete.")
