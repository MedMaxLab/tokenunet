import os
import sys
import json
import importlib
import inspect
from pathlib import Path
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# ─────────────────────────────────────────────────────────────────────────────
# 1. USER CONFIGURATION: Define your paths and trainers here
# ─────────────────────────────────────────────────────────────────────────────
TRAINERS_TO_CONSIDER = [
    "nnUNet_NDSnnUNetTrainer_100epochs",
]

NNUNET_ROOT         = Path("/home/tshimanga/Repositories/nnUNet")
NNUNET_RESULTS      = Path(os.environ.get("nnUNet_results", "data/tshimanga/nnUNet_results"))
NNUNET_PREPROCESSED = Path(os.environ.get("nnUNet_preprocessed", "/data/tshimanga/nnUNet_preprocessed"))

DATASET_NAME  = "Dataset001_FeTS"
PLANS_NAME    = "nnUNetPlans"
CONFIGURATION = "3d_fullres"
FOLD          = 0
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CUSTOM_MODULE_PATH = NNUNET_ROOT / "nnunetv2" / "training" / "nnUNetTrainer" / "custom" / "nnUNetTokenUNetTrainer.py"

# ─────────────────────────────────────────────────────────────────────────────
# 2. CORE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def discover_custom_trainers(module_path: Path) -> dict[str, type]:
    """Dynamically imports the trainer file and returns matching classes."""
    nnunet_src = str(module_path.parents[4])
    if nnunet_src not in sys.path:
        sys.path.insert(0, nnunet_src)

    spec = importlib.util.spec_from_file_location("custom_trainers", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return {
        name: obj for name, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, nnUNetTrainer) and obj is not nnUNetTrainer and obj.__module__ == module.__name__
    }

def load_plans(trainer_name: str) -> dict:
    plans_file = NNUNET_RESULTS / DATASET_NAME / f"{trainer_name}__{PLANS_NAME}__{CONFIGURATION}" / "plans.json"
    with open(plans_file) as f:
        return json.load(f)

def load_dataset_json() -> dict:
    with open(NNUNET_PREPROCESSED / DATASET_NAME / "dataset.json") as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# 3. EXECUTION FLOW
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dataset_json = load_dataset_json()
    all_discovered = discover_custom_trainers(CUSTOM_MODULE_PATH)
    
    print(f"\n{'─'*45}\n Trainer Parameter Counts\n{'─'*45}")
    
    for name in TRAINERS_TO_CONSIDER:
        if name not in all_discovered:
            print(f" ✗ {name}: Class not found in module.")
            continue
            
        try:
            plans = load_plans(name)
            plans["continue_training"] = False
            
            # Instantiate and initialize network structure via standard nnUNet routines
            trainer = all_discovered[name](
                plans=plans, configuration=CONFIGURATION, fold=FOLD, 
                dataset_json=dataset_json, device=DEVICE
            )
            trainer.initialize()
            
            # Count and print metrics
            n_params = sum(p.numel() for p in trainer.network.parameters()) / 1e6
            print(f" ✓ {name:<45} : {n_params:.2f} M")
            
            # Clean memory immediately
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f" ✗ {name}: Failed to resolve footprint due to: {e}")
            
    print(f"{'─'*45}")