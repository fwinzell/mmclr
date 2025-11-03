from datasets import MRIDataset
import torch
import os
import numpy as np


main_dir = "/Volumes/PA_CPGARCHIVE/projects/chimera/_aws/task1/radiology/images"
dataset = MRIDataset(radiology_dir=main_dir)
print(f"Number of patients in dataset: {len(dataset)}")






