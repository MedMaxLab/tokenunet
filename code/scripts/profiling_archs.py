"""
nnUNetv2 — Multi-Trainer Architecture Profiling
================================================
Discovers all custom trainer classes from a local module, builds each
network via its own build_network_architecture(), loads fold-0 weights,
and profiles inference + training throughput.

Output: a summary dict (and optional CSV) with per-trainer results.

Directory assumptions
---------------------
Custom trainers live in:
  <NNUNET_ROOT>/nnunetv2/training/nnUNetTrainer/custom/nnUNetTokenUNetTrainer.py

Results are stored as:
  $nnUNet_results/<DATASET_NAME>/<TrainerName>__<PLANS_NAME>__<CONFIGURATION>/
      plans.json
      fold_0/
          checkpoint_best.pth
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, json, time, pickle, importlib, inspect, contextlib, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

# 1. Hardware Mask: Force the script to only see GPU 3
# (Do this BEFORE importing or initializing CUDA)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 2. Performance Boost: Enable Tensor Cores for Ampere+ GPUs
torch.set_float32_matmul_precision('high')


import torch.distributed as dist

# 3. Software Bypass: Trick nnUNet into thinking DDP is running on a 1-GPU cluster
if not dist.is_initialized():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='gloo', rank=0, world_size=1)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION  ← edit these
# ─────────────────────────────────────────────────────────────────────────────

# Root of the nnUNetv2 source tree (the directory that contains nnunetv2/)
NNUNET_ROOT         = Path("/home/tshimanga/Repositories/nnUNet")

NNUNET_RESULTS      = Path(os.environ.get("nnUNet_results",      "data/tshimanga/nnUNet_results"))
NNUNET_PREPROCESSED = Path(os.environ.get("nnUNet_preprocessed", "/data/tshimanga/nnUNet_preprocessed"))

DATASET_NAME  = "Dataset001_FeTS"
PLANS_NAME    = "nnUNetPlans"
CONFIGURATION = "3d_fullres"
FOLD          = 0

# Profiling settings
N_WARMUP_INF   = 2    # warm-up passes before timing inference
N_RUNS_INF     = 10   # timed inference passes
N_WARMUP_TRAIN = 2    # warm-up passes before timing training step
N_RUNS_TRAIN   = 5    # timed training passes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Where to write the CSV summary (set to None to skip)
OUTPUT_CSV = Path("/home/tshimanga/Repositories/tokenunet/outputs/profiling_results.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DISCOVER CUSTOM TRAINERS
#     Imports the custom module and returns every class that:
#       • is defined in that module (not just re-imported)
#       • is a strict subclass of nnUNetTrainer
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_MODULE_PATH = (
    NNUNET_ROOT
    / "nnunetv2" / "training" / "nnUNetTrainer"
    / "custom" / "nnUNetTokenUNetTrainer.py"
)


def discover_custom_trainers(module_path: Path = CUSTOM_MODULE_PATH) -> dict[str, type]:
    """
    Dynamically import the custom trainer module and return
    {class_name: class} for every nnUNetTrainer subclass defined in it.
    """
    # Make sure the nnUNet source root is on sys.path so relative imports work
    nnunet_src = str(module_path.parents[4])   # four levels up from the .py = repo root
    if nnunet_src not in sys.path:
        sys.path.insert(0, nnunet_src)

    spec   = importlib.util.spec_from_file_location("custom_trainers", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    trainers = {}
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, nnUNetTrainer)
            and obj is not nnUNetTrainer
            and obj.__module__ == module.__name__   # defined here, not just imported
        ):
            trainers[name] = obj

    print(f"Discovered {len(trainers)} custom trainer(s):")
    for name in trainers:
        print(f"  - {name}")
    trainers_copy = trainers.copy()
    for trainer_name, trainer_cls in trainers_copy.items():
        if "NDSnnUNetTrainer" in trainer_name:
            print(f"\n  → Skipping {trainer_name} (Blacklisted due to DDP Deep Supervision conflict)\n")
            del trainers[trainer_name]
            continue

    print(f"Maintained {len(trainers)} custom trainer(s):")
    for name in trainers:
        print(f"  - {name}")
    return trainers


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LOAD PLANS AND DATASET JSON
# ─────────────────────────────────────────────────────────────────────────────

def load_plans_for_trainer(trainer_name: str) -> dict:
    """
    Each trainer has its own result folder (and therefore its own plans.json).
    We load from there to guarantee plans match the checkpoint exactly.
    """
    plans_file = (
        NNUNET_RESULTS
        / DATASET_NAME
        / f"{trainer_name}__{PLANS_NAME}__{CONFIGURATION}"
        / "plans.json"
    )
    if not plans_file.exists():
        raise FileNotFoundError(f"plans.json not found for {trainer_name}:\n  {plans_file}")
    with open(plans_file) as f:
        return json.load(f)

    if "continue_training" not in plans:
        plans["continue_training"] = False
    return plans

def load_dataset_json() -> dict:
    dataset_json = NNUNET_PREPROCESSED / DATASET_NAME / "dataset.json"
    with open(dataset_json) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BUILD NETWORK VIA TRAINER
#     Instantiating the trainer and calling build_network_architecture()
#     directly is the cleanest way to guarantee the custom build logic runs,
#     rather than re-implementing plans parsing here.
# ─────────────────────────────────────────────────────────────────────────────

def build_network_from_trainer(
    trainer_cls: type,
    plans:       dict,
    dataset_json: dict,
):
    """
    Construct a trainer instance (no training, no file I/O) and call
    initialize() so the network is built by the trainer's own
    build_network_architecture().  Returns the bare nn.Module.
    """
    plans["continue_training"] = False
    trainer = trainer_cls(
        plans        = plans,
        configuration= CONFIGURATION,
        fold         = FOLD,
        dataset_json = dataset_json,
        device       = DEVICE,
    )

    # initialize() builds self.network, self.optimizer, self.loss
    # It also calls set_deep_supervision_enabled() which custom trainers
    # may override (e.g. NDS trainer always sets it False).
    trainer.initialize()

    net = trainer.network
    net.to(DEVICE)
    return net


# ─────────────────────────────────────────────────────────────────────────────
# 5.  LOAD CHECKPOINT WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(net: nn.Module, trainer_name: str) -> nn.Module:
    ckpt_path = (
        NNUNET_RESULTS
        / DATASET_NAME
        / f"{trainer_name}__{PLANS_NAME}__{CONFIGURATION}"
        / f"fold_{FOLD}"
        / "checkpoint_best.pth"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n  {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = ckpt["network_weights"]
    
    # 1. Strip any unexpected 'module.' or '_orig_mod.' prefixes from saved checkpoint keys
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k
        if clean_key.startswith("module."):
            clean_key = clean_key[7:]
        if clean_key.startswith("_orig_mod."):
            clean_key = clean_key[10:]
        clean_state_dict[clean_key] = v

    # 2. Extract the actual raw target model underlying any combination of DDP or Compile wrappers
    raw_net = net
    while hasattr(raw_net, "module") or hasattr(raw_net, "_orig_mod"):
        if hasattr(raw_net, "module"):
            raw_net = raw_net.module
        if hasattr(raw_net, "_orig_mod"):
            raw_net = raw_net._orig_mod

    # 3. Securely load clean weights into the unwrapped raw model layers
    raw_net.load_state_dict(clean_state_dict)
        
    print(f"  Loaded weights safely (epoch {ckpt.get('current_epoch', '?')}): {ckpt_path.name}")
    return net


# ─────────────────────────────────────────────────────────────────────────────
# 6.  LOAD A PREPROCESSED CASE
# ─────────────────────────────────────────────────────────────────────────────

import blosc2  # <--- Added for reading .b2nd files

PREPROCESSED_DIR = NNUNET_PREPROCESSED / DATASET_NAME / f"{PLANS_NAME}_{CONFIGURATION}"


def list_cases() -> list[str]:
    return sorted(
        p.stem for p in PREPROCESSED_DIR.glob("*.b2nd")
        if not p.stem.endswith("_seg")
    )


def load_case(case_id: str):
    """
    Returns
    -------
    image : torch.Tensor  [1, C, X, Y, Z]
    label : torch.Tensor  [1, 1, X, Y, Z]  — extra dim for loss compatibility
    props : dict
    """
    # Use blosc2 to open the files and slice [:] to convert them to numpy arrays
    image = blosc2.open(str(PREPROCESSED_DIR / f"{case_id}.b2nd"))[:]        # [C,X,Y,Z]
    label = blosc2.open(str(PREPROCESSED_DIR / f"{case_id}_seg.b2nd"))[:]    # [X,Y,Z]
    
    with open(PREPROCESSED_DIR / f"{case_id}.pkl", "rb") as f:
        props = pickle.load(f)

    image_t = torch.from_numpy(image).unsqueeze(0).to(DEVICE)           # [1,C,X,Y,Z]
    label_t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).long().to(DEVICE)  # [1,1,X,Y,Z]
    return image_t, label_t, props


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PROFILING
# ─────────────────────────────────────────────────────────────────────────────

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _reset_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def _peak_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return float("nan")


def profile_inference(net: nn.Module, image: torch.Tensor) -> dict:
    """
    Pure forward pass, deep supervision off, no grad.
    Returns mean_ms, std_ms, peak_memory_mb.
    """
    # Turn off DS for inference regardless of training state
    ds_state = getattr(net, "deep_supervision", None)
    if ds_state is not None:
        net.deep_supervision = False

    net.eval()
    times = []

    with torch.no_grad():
        for _ in range(N_WARMUP_INF):
            net(image)
        _sync()
        _reset_mem()
        for _ in range(N_RUNS_INF):
            t0 = time.perf_counter()
            net(image)
            _sync()
            times.append((time.perf_counter() - t0) * 1000)

    result = dict(
        mean_ms        = float(np.mean(times)),
        std_ms         = float(np.std(times)),
        peak_memory_mb = _peak_mem_mb(),
    )

    # Restore DS state
    if ds_state is not None:
        net.deep_supervision = ds_state

    return result


def profile_training_step(net: nn.Module, image: torch.Tensor, label: torch.Tensor) -> dict:
    """
    Forward + backward + optimizer step, mirroring nnUNet training.
    Deep supervision is used only if the network supports it (ds_state check).
    Returns mean_ms, std_ms, peak_memory_mb.
    """
    ds_capable = hasattr(net, "deep_supervision")
    ds_state   = getattr(net, "deep_supervision", False)

    # Build loss: wrapped if DS is on, raw if DS is off or unsupported
    raw_loss = DC_and_CE_loss(
        {"batch_dice": True, "smooth": 1e-5, "do_bg": False},
        {},
        weight_ce=1.0,
        weight_dice=1.0,
        ignore_label=None,
    )
    use_ds = ds_capable and ds_state
    loss_fn = DeepSupervisionWrapper(raw_loss, weights=None) if use_ds else raw_loss

    optimizer = SGD(net.parameters(), lr=1e-2, momentum=0.99,
                    nesterov=True, weight_decay=3e-5)
    net.train()
    times = []

    def _step():
        # 1. Forward Pass
        out = net(image)
        
        # 2. Match nnUNet's Deep Supervision Target Layout
        if isinstance(out, (list, tuple)):
            # If the network outputted multiple scales (e.g., 3 to 5 scales depending on the layout)
            # Create a matching tuple where the label is formatted to match every downsampled target size.
            # Since we are just profiling step metrics, we can simply resize/interpolate the label 
            # to match the resolution of each scale outputted by the decoder.
            import torch.nn.functional as F
            
            labels_tuple = []
            for target_scale in out:
                target_shape = target_scale.shape[2:] # Get [X, Y, Z] of this specific head
                
                # Downsample or upscale our 128x128x128 label mask to match this head's spatial shape
                if list(label.shape[2:]) == list(target_shape):
                    labels_tuple.append(label)
                else:
                    downsampled_label = F.interpolate(
                        label, 
                        size=target_shape, 
                        mode="nearest" # Safe for binary region channels
                    )
                    labels_tuple.append(downsampled_label)
                    
            # Pass the structured list/tuple directly into nnUNet's native compound loss function
            loss = loss_fn(out, tuple(labels_tuple))
            
        else:
            # Standard single-output path (e.g., your TokenUNet models)
            loss = loss_fn(out, label)
            
        # 3. Backward Pass & Optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    for _ in range(N_WARMUP_TRAIN):
        _step()
    _sync()

    _reset_mem()
    for _ in range(N_RUNS_TRAIN):
        _sync()
        t0 = time.perf_counter()
        _step()
        _sync()
        times.append((time.perf_counter() - t0) * 1000)

    return dict(
        mean_ms        = float(np.mean(times)),
        std_ms         = float(np.std(times)),
        peak_memory_mb = _peak_mem_mb(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN LOOP — iterate over all discovered trainers
# ─────────────────────────────────────────────────────────────────────────────

def run_profiling(case_id: str | None = None):
    """
    For every discovered custom trainer:
      1. Load its plans.json  (from its own result folder)
      2. Build the network via trainer.initialize()
      3. Load fold-0 checkpoint weights
      4. Profile inference and training on one preprocessed case

    Parameters
    ----------
    case_id : str | None
        Specific case to use.  If None, the first available case is used.

    Returns
    -------
    results : {trainer_name: {inf: {...}, train: {...}}}
    """
    dataset_json = load_dataset_json()
    trainers     = discover_custom_trainers()

    # Pick a case once — same case for every architecture for fair comparison
    cases   = list_cases()
    case_id = case_id or cases[0]
    print(f"\nProfiling case: {case_id}\n{'─'*60}")
    image, label, props = load_case(case_id)
    print(f"  image {image.shape}  label {label.shape}  "
          f"spacing {props.get('spacing', 'N/A')}\n")
          
    # Extract the standard patch size used by normal trainers
    target_patch_size = [128,128,128] # e.g., [64, 128, 128]
    
    # Adjust image to match the trainer patch size (Pad if too small, Crop if too large)
    import torch.nn.functional as F
    
    # Work backward from the spatial dimensions (Z, Y, X)
    img_spatial_dims = [image.shape[-3], image.shape[-2], image.shape[-1]]
    
    pad_list = []
    for i in [2, 1, 0]:  # Z, Y, X
        curr = img_spatial_dims[i]
        target = target_patch_size[i]
        if curr < target:
            pad_total = target - curr
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            pad_list.extend([pad_left, pad_right])
        else:
            pad_list.extend([0, 0])
            
    # Pad both tensors safely
    image = F.pad(image, pad_list, mode="constant", value=0)
    label = F.pad(label, pad_list, mode="constant", value=0)
    
    # Crop backward from the spatial dimensions (X, Y, Z)
    for i in [0, 1, 2]:  # X, Y, Z
        img_axis = image.ndim - 3 + i
        lbl_axis = label.ndim - 3 + i
        
        curr = image.shape[img_axis]
        target = target_patch_size[i]
        if curr > target:
            start = (curr - target) // 2
            image = image.narrow(img_axis, start, target)
            label = label.narrow(lbl_axis, start, target)
            
    # ─────────────────────────────────────────────────────────────────────────
    # FIX: DYNAMICALLY FORMAT LABEL TO REGION-BASED CHANNELS FOR LOSS FUNCTION
    # ─────────────────────────────────────────────────────────────────────────
    print("  → Rebuilding label tensor layout into region-based target channels...")
    
    # Squeeze out potential leading batch/channel dimensions to isolate a clean 3D volume
    while label.ndim > 3:
        label = label.squeeze(0)
        
    # Standard labels map: 1: NCR, 2: ED, 3: ET
    # Construct the 3 target regions required by the nnUNet loss module
    region_et = (label == 3)
    region_tc = (label == 1) | (label == 3)
    region_wt = (label == 1) | (label == 2) | (label == 3)
    
    # Stack along the channel axis and re-add batch dimension -> [B=1, C=3, X, Y, Z]
    label = torch.stack([region_et, region_tc, region_wt], dim=0).unsqueeze(0).float()
    
    # Ensure image has a solid batch dimension as well -> [B=1, C=4, X, Y, Z]
    if image.ndim == 4:
        image = image.unsqueeze(0)
        
    print(f"✓ Adjusted inputs -> image: {image.shape} | label: {label.shape}")
    # ─────────────────────────────────────────────────────────────────────────

    results = {}

    for trainer_name, trainer_cls in trainers.items():
        if "SwinUNETR" in trainer_name:
            # Force nnUNet to drop to Eager mode for this specific model
            os.environ["nnUNet_compile"] = "f"
        else:
            # Let other models use torch.compile as normal
            os.environ["nnUNet_compile"] = "t"
        print(f"{'═'*60}")
        print(f"  Trainer : {trainer_name}")
        print(f"{'═'*60}")

        try:
            plans = load_plans_for_trainer(trainer_name)
        except FileNotFoundError as e:
            print(f"  ✗ Skipping — {e}\n")
            continue

        try:
            net = build_network_from_trainer(trainer_cls, plans, dataset_json)
            
            # ─────────────────────────────────────────────────────────────────
            # FIX: FORCE DDP TO IGNORE UNUSED DEEP SUPERVISION PARAMETERS
            # ─────────────────────────────────────────────────────────────────
            if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                net.find_unused_parameters = True
                
        except Exception as e:
            print(f"  ✗ Network build failed! Full traceback below:\n")
            import traceback
            traceback.print_exc()  
            print("-" * 60 + "\n")
            continue

        try:
            net = load_checkpoint(net, trainer_name)
        except FileNotFoundError as e:
            print(f"  ✗ Checkpoint missing — {e}")
            print(    "    Continuing with random weights.\n")

        n_params = sum(p.numel() for p in net.parameters()) / 1e6
        print(f"  Parameters : {n_params:.2f} M")

        print("  → Profiling inference ...")
        inf_stats   = profile_inference(net, image)

        print("  → Profiling training step ...")
        train_stats = profile_training_step(net, image, label)

        results[trainer_name] = {"inference": inf_stats, "training": train_stats, "n_params": n_params}

        print(f"  Inference  : {inf_stats['mean_ms']:.1f} ± {inf_stats['std_ms']:.1f} ms"
              f"   peak {inf_stats['peak_memory_mb']:.0f} MB")
        print(f"  Training   : {train_stats['mean_ms']:.1f} ± {train_stats['std_ms']:.1f} ms"
              f"   peak {train_stats['peak_memory_mb']:.0f} MB\n")

        # Free GPU memory before next model
        del net
        torch.cuda.empty_cache()

    return results


def save_csv(results: dict[str, dict], path: Path = OUTPUT_CSV):
    rows = []
    for trainer_name, stats in results.items():
        rows.append({
            "trainer"              : trainer_name,
            "inf_mean_ms"          : stats["inference"]["mean_ms"],
            "inf_std_ms"           : stats["inference"]["std_ms"],
            "inf_peak_mem_mb"      : stats["inference"]["peak_memory_mb"],
            "train_mean_ms"        : stats["training"]["mean_ms"],
            "train_std_ms"         : stats["training"]["std_ms"],
            "train_peak_mem_mb"    : stats["training"]["peak_memory_mb"],
            "n_params"             : stats["n_params"]
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_profiling(case_id=None)   # pass a case ID string to fix the case

    print("\n" + "═"*60)
    print("SUMMARY")
    print("═"*60)
    for trainer_name, stats in results.items():
        i, t = stats["inference"], stats["training"]
        print(f"{trainer_name}")
        print(f"  Inference : {i['mean_ms']:7.1f} ± {i['std_ms']:.1f} ms  |  {i['peak_memory_mb']:.0f} MB")
        print(f"  Training  : {t['mean_ms']:7.1f} ± {t['std_ms']:.1f} ms  |  {t['peak_memory_mb']:.0f} MB")

    if OUTPUT_CSV and results:
        save_csv(results)