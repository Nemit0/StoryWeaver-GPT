import kagglehub
import os

# Download latest version of the dataset
os.mkdir("./data_test")
path = kagglehub.dataset_download("ratthachat/writing-prompts")
print("Path to dataset files:", path)