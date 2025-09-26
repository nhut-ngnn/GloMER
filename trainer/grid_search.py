import os
import numpy as np
import subprocess
from tqdm import tqdm

alphas = np.round(np.arange(0.0, 1.51, 0.1), 2)


data_dir = "features_output/"           
dataset = "ESD"                  
num_classes = 5                     
epochs = 100

os.makedirs("grid_search_logs", exist_ok=True)

for alpha in alphas:
    print(f"\n=== Running Grid Search for alpha = {alpha} ===")

    log_file = f"grid_search_logs/{dataset}_{num_classes}class_alpha{alpha:.2f}.log"

    cmd = [
        "python", "trainer/train.py",
        "--data_dir", data_dir,
        "--dataset", dataset,
        "--num_classes", str(num_classes),
        "--epochs", str(epochs),
        "--alpha", str(alpha)
    ]

    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        process.wait()

    print(f"Finished alpha = {alpha}, log saved to {log_file}")

print("\n=== Grid Search Completed ===")
