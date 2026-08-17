"""
nnUNetv2 — Token Attention Map Alignment & Visualization
======================================================
Evaluates selected TokenUNet architectures across multiple folds.
Uses PyTorch forward hooks to extract spatial attention maps from
the TokenLearner and TokenFuser modules, interpolates them to the 
original resolution, and computes Soft Dice against the 3 target 
regions (ET, TC, WT).

Finally, it aggregates the Max and Mean Dice scores across folds
and plots a mid-tumor slice for visual inspection.
"""

import os, sys, json, pickle, importlib, inspect, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import blosc2

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# ─────────────────────────────────────────────────────────────────────────────
# 0.  ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.set_float32_matmul_precision('high')

import torch.distributed as dist
if not dist.is_initialized():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='gloo', rank=0, world_size=1)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
NNUNET_ROOT         = Path("/home/tshimanga/Repositories/nnUNet")
NNUNET_RESULTS      = Path(os.environ.get("nnUNet_results", "data/tshimanga/nnUNet_results"))
NNUNET_PREPROCESSED = Path(os.environ.get("nnUNet_preprocessed", "/data/tshimanga/nnUNet_preprocessed"))

DATASET_NAME  = "Dataset001_FeTS"
PLANS_NAME    = "nnUNetPlans"
CONFIGURATION = "3d_fullres"

# Evaluate across these folds
FOLDS = [0, 1, 2, 3, 4]

# ONLY evaluate these specific architectures (List your specific TokenUNets here)
TARGET_TRAINERS = [
    "nnUNet_8TokenUNetTrainer_100epochs",
    "nnUNet_8AttnTokenUNetTrainer_100epochs",
    "nnUNet_8MLPTokenUNetTrainer_100epochs",
    "nnUNet_32TokenUNetTrainer_100epochs",
    "nnUNet_32AttnTokenUNetTrainer_100epochs",
    "nnUNet_32AttnLongTokenUNetTrainer_100epochs",
    "nnUNet_32MLPTokenUNetTrainer_100epochs" 
]

# Provide the exact internal module paths to your Learner and Fuser
# (You may need to adjust these based on your specific implementation)
HOOK_LAYERS = {
    "TokenLearner": "token_learner.mask_maker",
    "TokenFuser":   "token_fuser.Beta"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("/home/tshimanga/Repositories/tokenunet/outputs/attention_alignment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FORWARD HOOK EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
class ActivationExtractor:
    """Attaches to a specific layer to grab its output during the forward pass."""
    def __init__(self, model, layer_name):
        self.activation = None
        self.hook = None
        
        # Traverse the model to find the specific layer
        layer = model
        for part in layer_name.split('.'):
            if hasattr(layer, part):
                layer = getattr(layer, part)
            else:
                print(f"  [Warning] Layer '{layer_name}' not found. Check HOOK_LAYERS config.")
                return
                
        self.hook = layer.register_forward_hook(self._hook_fn)
        
    def _hook_fn(self, module, input, output):
        # Handle cases where the module returns a tuple (take the first tensor)
        if isinstance(output, tuple):
            self.activation = output[0].detach()
        else:
            self.activation = output.detach()
            
    def remove(self):
        if self.hook is not None:
            self.hook.remove()

# ─────────────────────────────────────────────────────────────────────────────
# 3.  MATH & METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_soft_dice(pred_map: torch.Tensor, target_mask: torch.Tensor, smooth: float = 1e-5):
    """
    Computes Soft Dice between an attention map [0,1] and a binary target mask.
    pred_map: [X, Y, Z]
    target_mask: [X, Y, Z]
    """
    # Ensure map is strictly between 0 and 1
    pred_map = torch.clamp(pred_map, 0.0, 1.0)
    
    intersection = torch.sum(pred_map * target_mask)
    pred_sum = torch.sum(pred_map)
    target_sum = torch.sum(target_mask)
    
    return (2. * intersection + smooth) / (pred_sum + target_sum + smooth)

def normalize_map(m: torch.Tensor):
    """Min-Max normalization per spatial map to ensure a 0-1 range for soft Dice."""
    m_min = m.min()
    m_max = m.max()
    if m_max - m_min > 1e-6:
        return (m - m_min) / (m_max - m_min)
    return m

# ─────────────────────────────────────────────────────────────────────────────
# 4.  DATA & MODEL LOADING (Reused from your script)
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset_json():
    with open(NNUNET_PREPROCESSED / DATASET_NAME / "dataset.json") as f:
        return json.load(f)

def load_case(case_id: str):
    preproc_dir = NNUNET_PREPROCESSED / DATASET_NAME / f"{PLANS_NAME}_{CONFIGURATION}"
    image = blosc2.open(str(preproc_dir / f"{case_id}.b2nd"))[:]
    label = blosc2.open(str(preproc_dir / f"{case_id}_seg.b2nd"))[:]
    image_t = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    label_t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).long().to(DEVICE)
    return image_t, label_t

def process_maps(activation: torch.Tensor, labels: torch.Tensor, label_names: list):
    """Upsamples the extracted attention maps and computes Soft Dice against GT labels."""
    if activation is None:
        return None, {lbl: {"max": 0.0, "avg": 0.0} for lbl in label_names}
        
    # 1. Upsample [B, N, X_small, Y_small, Z_small] -> [B, N, X_tgt, Y_tgt, Z_tgt]
    target_shape = labels.shape[2:]
    upsampled = F.interpolate(activation, size=target_shape, mode='trilinear', align_corners=False)
    
    N_tokens = upsampled.shape[1]
    metrics = {lbl: {"max": 0.0, "avg": 0.0} for lbl in label_names}
    
    # 2. Compute Dice for each target region
    for lbl_idx, lbl_name in enumerate(label_names):
        target_mask = labels[0, lbl_idx]
        token_scores = []
        
        for t in range(N_tokens):
            token_map = normalize_map(upsampled[0, t])
            score = compute_soft_dice(token_map, target_mask)
            token_scores.append(score.item())
            
        metrics[lbl_name]["max"] = max(token_scores) if token_scores else 0.0
        metrics[lbl_name]["avg"] = sum(token_scores) / N_tokens if N_tokens > 0 else 0.0
        
    return upsampled, metrics

def format_data_for_eval(image, label, target_patch=[128,128,128]):
    img_spatial = [image.shape[-3], image.shape[-2], image.shape[-1]]
    pad_list = []
    for i in [2, 1, 0]:
        curr, tgt = img_spatial[i], target_patch[i]
        if curr < tgt:
            pad = tgt - curr
            pad_list.extend([pad // 2, pad - (pad // 2)])
        else:
            pad_list.extend([0, 0])
            
    image = F.pad(image, pad_list, mode="constant", value=0)
    label = F.pad(label, pad_list, mode="constant", value=0)
    
    for i in [0, 1, 2]:
        img_axis, lbl_axis = image.ndim - 3 + i, label.ndim - 3 + i
        curr, tgt = image.shape[img_axis], target_patch[i]
        if curr > tgt:
            image = image.narrow(img_axis, (curr - tgt) // 2, tgt)
            label = label.narrow(lbl_axis, (curr - tgt) // 2, tgt)
            
    while label.ndim > 3: label = label.squeeze(0)
    
    # 1: NCR, 2: ED, 3: ET -> Build ET, TC, WT
    region_et = (label == 3)
    region_tc = (label == 1) | (label == 3)
    region_wt = (label == 1) | (label == 2) | (label == 3)
    
    label_regions = torch.stack([region_et, region_tc, region_wt], dim=0).unsqueeze(0).float()
    if image.ndim == 4: image = image.unsqueeze(0)
    
    return image, label_regions

# ─────────────────────────────────────────────────────────────────────────────
# 5.  VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(case_id, trainer_name, fold, image, labels, tl_maps, tf_maps, slice_idx):
    """Plots the image, GT regions, and top Token maps for a specific Z-slice."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f"{trainer_name} - Fold {fold} - Case {case_id} (Z={slice_idx})", fontsize=16)
    
    # Row 0: Inputs
    axes[0, 0].imshow(image[0, 0, :, :, slice_idx].cpu(), cmap="gray")
    axes[0, 0].set_title("Input (T1c/Modality 0)")
    axes[0, 1].imshow(labels[0, 0, :, :, slice_idx].cpu(), cmap="Reds")
    axes[0, 1].set_title("GT: Enhancing Tumor (ET)")
    axes[0, 2].imshow(labels[0, 1, :, :, slice_idx].cpu(), cmap="Greens")
    axes[0, 2].set_title("GT: Tumor Core (TC)")
    axes[0, 3].imshow(labels[0, 2, :, :, slice_idx].cpu(), cmap="Blues")
    axes[0, 3].set_title("GT: Whole Tumor (WT)")
    
    # Helper to plot maps intelligently
    def plot_maps(row, name, maps):
        axes[row, 0].axis('off')
        axes[row, 0].text(0.5, 0.5, f"{name}\nAttention Maps", ha='center', va='center', fontsize=14)
        
        label_names = ["ET", "TC", "WT"]
        
        # Iterate over the 3 GT labels to find the best token for each
        for i, lbl_name in enumerate(label_names):
            target_mask = labels[0, i] # The 3D GT mask for this label
            
            best_token_idx = 0
            best_dice = -1.0
            
            for t in range(maps.shape[1]):
                # Use the exact same Dice calculation as the logging loop
                token_map = normalize_map(maps[0, t])
                score = compute_soft_dice(token_map, target_mask).item()
                
                if score > best_dice:
                    best_dice = score
                    best_token_idx = t
                    
            # Plot that specific best token and add the Dice score to the title
            axes[row, i+1].imshow(maps[0, best_token_idx, :, :, slice_idx].cpu(), cmap="inferno")
            axes[row, i+1].set_title(f"{lbl_name} Match (Token {best_token_idx})\nDice: {best_dice:.3f}")

    if tl_maps is not None: plot_maps(1, "TokenLearner", tl_maps)
    if tf_maps is not None: plot_maps(2, "TokenFuser", tf_maps)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{trainer_name}_F{fold}_{case_id}.png")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    dataset_json = load_dataset_json()
    # Load the nnUNet data splits
    splits_file = NNUNET_PREPROCESSED / DATASET_NAME / "splits_final.json"
    with open(splits_file, "r") as f:
        splits = json.load(f)
    
    # Use your discover function, but filter for TARGET_TRAINERS
    module_path = NNUNET_ROOT / "nnunetv2" / "training" / "nnUNetTrainer" / "custom" / "nnUNetTokenUNetTrainer.py"
    spec = importlib.util.spec_from_file_location("custom_trainers", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    all_trainers = {name: obj for name, obj in inspect.getmembers(module, inspect.isclass) 
                    if issubclass(obj, nnUNetTrainer) and name in TARGET_TRAINERS}

    # Load 1 test case
    preproc_dir = NNUNET_PREPROCESSED / DATASET_NAME / f"{PLANS_NAME}_{CONFIGURATION}"
    cases = sorted(p.stem for p in preproc_dir.glob("*.b2nd") if not p.stem.endswith("_seg"))
    case_id = cases[0] 
    
    raw_img, raw_lbl = load_case(case_id)
    image, labels = format_data_for_eval(raw_img, raw_lbl)
    
    # Find the slice with the largest Whole Tumor (label idx 2) for visualization
    wt_sum_per_slice = labels[0, 2].sum(dim=(0, 1))
    best_slice_idx = int(torch.argmax(wt_sum_per_slice).item())
    
    label_names = ["ET", "TC", "WT"]
    results = {}

    for trainer_name, trainer_cls in all_trainers.items():
        results[trainer_name] = {}
        os.environ["nnUNet_compile"] = "f" # Always Eager for map extraction
        
        for fold in FOLDS:
            print(f"\nEvaluating {trainer_name} - Fold {fold}")
            
            plans_file = NNUNET_RESULTS / DATASET_NAME / f"{trainer_name}__{PLANS_NAME}__{CONFIGURATION}" / "plans.json"
            if not plans_file.exists():
                print(f"  ✗ Plans not found. Skipping fold.")
                continue
                
            with open(plans_file) as f: plans = json.load(f)
            plans["continue_training"] = False
            
            # Build
            trainer = trainer_cls(plans=plans, configuration=CONFIGURATION, fold=fold, dataset_json=dataset_json, device=DEVICE)
            trainer.initialize()
            net = trainer.network.to(DEVICE)
            
            # Load Weights
            ckpt_path = NNUNET_RESULTS / DATASET_NAME / f"{trainer_name}__{PLANS_NAME}__{CONFIGURATION}" / f"fold_{fold}" / "checkpoint_best.pth"
            if not ckpt_path.exists():
                print(f"  ✗ Checkpoint missing. Skipping fold.")
                continue
                
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            
            # 1. Clean the saved weights (strip prefixes)
            clean_state_dict = {
                k.replace("module.", "").replace("_orig_mod.", ""): v 
                for k, v in ckpt["network_weights"].items()
            }
            
            # 2. Extract the raw model (unwrap from DDP / Compile)
            raw_net = net
            while hasattr(raw_net, "module") or hasattr(raw_net, "_orig_mod"):
                if hasattr(raw_net, "module"):
                    raw_net = raw_net.module
                if hasattr(raw_net, "_orig_mod"):
                    raw_net = raw_net._orig_mod
                    
            # 3. Load cleanly into the raw underlying model
            raw_net.load_state_dict(clean_state_dict)
            raw_net.eval()

            # Attach Hooks directly to the unwrapped raw network
            ext_tl = ActivationExtractor(raw_net, HOOK_LAYERS["TokenLearner"])
            ext_tf = ActivationExtractor(raw_net, HOOK_LAYERS["TokenFuser"])

            # ─────────────────────────────────────────────────────────────
            # NEW: Loop over all validation cases for this specific fold
            # ─────────────────────────────────────────────────────────────
            val_cases = splits[fold]["val"]
            print(f"  → Processing {len(val_cases)} validation cases...")
            
            fold_tl_max_scores = {lbl: [] for lbl in label_names}
            fold_tl_avg_scores = {lbl: [] for lbl in label_names}
            fold_tf_max_scores = {lbl: [] for lbl in label_names}
            fold_tf_avg_scores = {lbl: [] for lbl in label_names}

            for case_idx, case_id in enumerate(val_cases):
                raw_img, raw_lbl = load_case(case_id)
                image, labels = format_data_for_eval(raw_img, raw_lbl)

                with torch.no_grad():
                    raw_net(image)

                # Process maps
                tl_maps_up, tl_metrics = process_maps(ext_tl.activation, labels, label_names)
                tf_maps_up, tf_metrics = process_maps(ext_tf.activation, labels, label_names)

                # Accumulate scores for this case
                for lbl in label_names:
                    fold_tl_max_scores[lbl].append(tl_metrics[lbl]["max"])
                    fold_tl_avg_scores[lbl].append(tl_metrics[lbl]["avg"])
                    fold_tf_max_scores[lbl].append(tf_metrics[lbl]["max"])
                    fold_tf_avg_scores[lbl].append(tf_metrics[lbl]["avg"])

                # Optional: Only plot the first validation case of the fold to save space
                if case_idx == 0:
                    wt_sum_per_slice = labels[0, 2].sum(dim=(0, 1))
                    best_slice_idx = int(torch.argmax(wt_sum_per_slice).item())
                    plot_results(case_id, trainer_name, fold, image, labels, tl_maps_up, tf_maps_up, best_slice_idx)

            # ─────────────────────────────────────────────────────────────
            # Aggregate validation scores for this fold
            # ─────────────────────────────────────────────────────────────
            fold_tl_metrics = {
                lbl: {
                    "max": sum(fold_tl_max_scores[lbl]) / len(val_cases),
                    "avg": sum(fold_tl_avg_scores[lbl]) / len(val_cases)
                } for lbl in label_names
            }
            
            fold_tf_metrics = {
                lbl: {
                    "max": sum(fold_tf_max_scores[lbl]) / len(val_cases),
                    "avg": sum(fold_tf_avg_scores[lbl]) / len(val_cases)
                } for lbl in label_names
            }

            results[trainer_name][fold] = {"TL": fold_tl_metrics, "TF": fold_tf_metrics}
            
            print(f"  Fold TL Dice (Mean of Max) -> ET: {fold_tl_metrics['ET']['max']:.3f} | TC: {fold_tl_metrics['TC']['max']:.3f} | WT: {fold_tl_metrics['WT']['max']:.3f}")
            print(f"  Fold TF Dice (Mean of Max) -> ET: {fold_tf_metrics['ET']['max']:.3f} | TC: {fold_tf_metrics['TC']['max']:.3f} | WT: {fold_tf_metrics['WT']['max']:.3f}")

            # Cleanup
            ext_tl.remove()
            ext_tf.remove()
            del net, trainer, raw_net
            torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────────────────
    # 7.  AGGREGATE, PRINT, AND SAVE SUMMARY
    # ─────────────────────────────────────────────────────────────────────────────
    import csv
    print("\n" + "═"*80)
    print("CROSS-FOLD ALIGNMENT SUMMARY")
    print("═"*80)
    
    # List to hold the rows for our CSV table
    table_data = []
    
    for trainer_name, folds_data in results.items():
        if not folds_data: continue
        
        print(f"\nArchitecture: {trainer_name} (Over {len(folds_data)} folds)")
        
        # Initialize the row for this specific architecture
        row_data = {"Architecture": trainer_name}
        
        for module_key in ["TL", "TF"]:
            print(f"  {module_key} Alignment (Average across folds):")
            for lbl in label_names:
                max_scores = [folds_data[f][module_key][lbl]["max"] for f in folds_data if folds_data[f][module_key]]
                avg_scores = [folds_data[f][module_key][lbl]["avg"] for f in folds_data if folds_data[f][module_key]]
                
                if max_scores:
                    mean_of_max = sum(max_scores) / len(max_scores)
                    mean_of_avg = sum(avg_scores) / len(avg_scores)
                    print(f"    - {lbl}: Best Token Dice = {mean_of_max:.3f} | All Tokens Avg = {mean_of_avg:.3f}")
                    
                    # Store the rounded metrics in our row dictionary
                    row_data[f"{module_key}_{lbl}_Best"] = round(mean_of_max, 3)
                    row_data[f"{module_key}_{lbl}_Avg"] = round(mean_of_avg, 3)
        
        table_data.append(row_data)

    # Save to CSV
    csv_file_path = OUTPUT_DIR / "alignment_metrics_summary.csv"
    
    if table_data:
        # Dynamically grab the column headers from the first dictionary
        headers = ["Architecture"] + [k for k in table_data[0].keys() if k != "Architecture"]
        
        with open(csv_file_path, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(table_data)
            
        print(f"\n[+] Summary successfully saved to {csv_file_path}")
        
if __name__ == "__main__":
    main()