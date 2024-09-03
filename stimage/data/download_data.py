import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the token from the environment
hf_token = os.getenv('HF_TOKEN')

if hf_token is None:
    raise ValueError("Hugging Face token is not found in .env file")

# Repository information
repo_id = 'jiawennnn/STimage-1K4M'  # Replace with the actual repository path
local_dir = '/workspaces/stimage/data/raw'  # Local folder where to save the files

# Download the repository using the token
snapshot_download(repo_id=repo_id, local_dir=local_dir, token=hf_token, repo_type="dataset")

print(f"Repository {repo_id} has been downloaded to {local_dir}")
