import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import importlib_resources as impresources
import json
import nltk
import subprocess
import sys
import os
import gc

def setup_resources():
    """
    Downloads the necessary NLTK packages.
    """
    print("Setting up DP-MLM resources...")
    
    # NLTK requirements
    nltk_packages = ['words', 'punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']
    
    for package in nltk_packages:
        nltk.download(package, quiet=True)
    print("✓ NLTK packages ready.")

    print("Setup complete.")

def calculate_logit_bounds(model_name, batch_size=16, percentile=0.01, device=None, bound_strategy=None):
    """
    Analyzes a model's logit distribution across raw text to determine DP clipping bounds.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    with open(impresources.files("dpmlm") / "data" / "calibration.txt", 'r') as f:
        calibration_texts = json.load(f)

    all_logits = []

    for i in tqdm(range(0, len(calibration_texts), batch_size), desc="Measuring Logits"):
        batch = calibration_texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            special_tokens_mask = (inputs['input_ids'] == tokenizer.pad_token_id) | \
                                 (inputs['input_ids'] == tokenizer.cls_token_id) | \
                                 (inputs['input_ids'] == tokenizer.sep_token_id)
            content_logits = logits[~special_tokens_mask] 
            
            all_logits.append(content_logits.cpu().numpy().flatten())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    flattened = np.concatenate(all_logits)
    
    stats = {
        "mean": np.mean(flattened),
        "std": np.std(flattened),
        "q_low": np.percentile(flattened, percentile),
        "q_high": np.percentile(flattened, 1-percentile)
    }

    if bound_strategy:
        clip_min, clip_max = bound_strategy(stats["mean"], stats["std"], stats["q_low"], stats["q_high"])
    else:
        # default is min and max
        clip_min, clip_max = stats["q_high"], stats["q_high"]

    return {
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
        "sensitivity": float(abs(clip_max - clip_min)),
        "raw_stats": stats
    }

def cleanup(model_instances=None):
    """
    Clears GPU memory and performs garbage collection.
    :param model_instances: Optional list of objects (DP-MLM instances) to delete.
    """
    print("Cleaning up...")

    if model_instances:
        if not isinstance(model_instances, list):
            model_instances = [model_instances]
        for obj in model_instances:
            # We target the actual heavy attributes
            if hasattr(obj, 'lm_model'):
                del obj.lm_model
            if hasattr(obj, 'ipi_pipe'):
                del obj.ipi_pipe
            del obj
            
    gc.collect()
    
    if torch.cuda.is_available():
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    gc.collect()
            
    print("✓ GPU VRAM cleared and garbage collected.")