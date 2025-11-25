#!/usr/bin/env python3
"""
Processing Module

Contains the core logic for running segmentation inference and case processing.
"""
import torch
import numpy as np
import functools
from pathlib import Path
from typing import List, Dict, Any

# Assumes models.py, feature_extraction.py, and dataset.py are in the same package
from .unet_models import create_network_from_params
from .feature_extraction import run_ensemble_feature_extraction

#from .dataset import (
#    load_and_preprocess_case,
#    postprocess_detection_map,
#    save_features,
#    save_detection_map
#)
from datasets import MRIDataset

def process_single_case(
    case_id: str,
    input_files: List[str],
    plans: Dict[str, Any],
    network_params: Dict[str, Any],
    model_folder: Path,
    output_folder: Path,
    extract_features: bool,
    folds_to_use: List[int],
    probability_threshold: float
) -> bool:
    """
    Process a single case for either feature extraction or detection map generation.
    Returns True if successful, False otherwise.
    """
    try:
        print(f"📋 Processing case: {case_id}")
        
        target_patch_size = plans['plans_per_stage'][0]['patch_size']
        input_tensor, original_scan, original_shape = load_and_preprocess_case(
            case_id, input_files, target_patch_size
        )
        
        print("   💾 Processing and saving output...")
        ensemble_result = run_ensemble_feature_extraction(
            network_params, model_folder, input_tensor, folds_to_use
        )
        save_features(ensemble_result, case_id, output_folder)
        
        print(f"   ✅ Case {case_id} completed successfully!")
        return True
        
    except Exception as e:
        import traceback
        print(f"   ❌ Error processing case {case_id}: {e}")
        traceback.print_exc()
        return False