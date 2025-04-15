"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import numpy as np

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

class Config:
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = "section1/out"  # This is where images/labels live
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "section2/out/results"

if __name__ == "__main__":
    # Get configuration
    c = Config()

    # Ensure output directory exists
    os.makedirs(c.test_results_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    data = LoadHippocampusData(c.root_dir, y_shape=c.patch_size, z_shape=c.patch_size)

    # Create test-train-val split
    keys = np.arange(len(data))
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(keys)

    # 80% train, 10% val, 10% test
    n = len(keys)
    split = {
        "train": keys[:int(0.8 * n)].tolist(),
        "val": keys[int(0.8 * n):int(0.9 * n)].tolist(),
        "test": keys[int(0.9 * n):].tolist()
    }

    # Set up and run experiment
    exp = UNetExperiment(c, split, data)

    # Run training
    exp.run()

    # Run testing
    results_json = exp.run_test()

    # Save results and configuration
    results_json["config"] = vars(c)
    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))


