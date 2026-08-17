"""
FeTS (MICCAI BraTS legacy format) → nnUNet v2 format converter.

FeTS source layout (per subject):
    <fets_root>/
    └── <SubjectID>/                      # e.g. FeTS2022_00001
        ├── <SubjectID>_t1.nii.gz
        ├── <SubjectID>_t1ce.nii.gz
        ├── <SubjectID>_t2.nii.gz
        ├── <SubjectID>_flair.nii.gz
        └── <SubjectID>_seg.nii.gz

nnUNet v2 target layout:
    <nnunet_raw>/
    └── Dataset<ID>_FeTS/
        ├── dataset.json
        ├── imagesTr/
        │   ├── FeTS_001_0000.nii.gz      # T1
        │   ├── FeTS_001_0001.nii.gz      # T1ce
        │   ├── FeTS_001_0002.nii.gz      # T2
        │   ├── FeTS_001_0003.nii.gz      # FLAIR
        │   └── ...
        ├── imagesTs/                     # optional held-out test set
        │   └── ...
        └── labelsTr/
            ├── FeTS_001.nii.gz
            └── ...

Label mapping (BraTS/FeTS legacy):
    0  → background
    1  → NCR  (Necrotic Core)
    2  → ED   (Edema)
    4  → ET   (Enhancing Tumour)          ← NOTE: no label 3 in BraTS
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional nibabel import — only needed for label sanity-check
# ---------------------------------------------------------------------------
try:
    import nibabel as nib
    _NIBABEL_AVAILABLE = True
except ImportError:
    _NIBABEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODALITY_SUFFIXES = {
    "_t1":    "0000",
    "_t1ce":  "0001",
    "_t2":    "0002",
    "_flair": "0003",
}

SEG_SUFFIX = "_seg"

BRATS_LABELS = {
    "background": 0,
    "NCR":        1,
    "ED":         2,
    "ET":         4,   # intentional gap — no label 3 in BraTS/FeTS
}

DATASET_JSON_TEMPLATE = {
    "channel_names": {
        "0": "T1",
        "1": "T1ce",
        "2": "T2",
        "3": "FLAIR",
    },
    "labels": BRATS_LABELS,
    "file_ending": ".nii.gz",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_subject_dirs(fets_root: Path) -> list[Path]:
    """
    Return sorted list of subject directories.
    Accepts both flat layout (fets_root/<SubjectID>/) and a single extra
    nesting level (fets_root/<split>/<SubjectID>/) as shipped by some FeTS
    data releases.
    """
    candidates = sorted(fets_root.iterdir())
    subject_dirs = []

    for c in candidates:
        if not c.is_dir():
            continue
        # Direct subject dir: contains at least one *_t1.nii.gz
        if any(c.glob("*_t1.nii.gz")):
            subject_dirs.append(c)
        else:
            # One level deeper (e.g. TrainingData/<SubjectID>)
            for sub in sorted(c.iterdir()):
                if sub.is_dir() and any(sub.glob("*_t1.nii.gz")):
                    subject_dirs.append(sub)

    return subject_dirs


def _resolve_file(subject_dir: Path, suffix: str) -> Optional[Path]:
    """
    Find a file ending with `suffix + '.nii.gz'` inside subject_dir.
    Case-insensitive on the suffix to handle minor naming inconsistencies.
    """
    pattern = f"*{suffix}.nii.gz"
    hits = list(subject_dir.glob(pattern))
    if not hits:
        # Try case-insensitive fallback
        hits = [f for f in subject_dir.glob("*.nii.gz")
                if f.name.lower().endswith(suffix.lower() + ".nii.gz")]
    return hits[0] if len(hits) == 1 else None


def _check_labels(seg_path: Path, expected: set[int]) -> list[int]:
    """
    Load segmentation and return any unexpected label values found.
    Requires nibabel.
    """
    if not _NIBABEL_AVAILABLE:
        return []
    img = nib.load(str(seg_path))
    data = np.asarray(img.dataobj, dtype=np.int16)
    unique = set(np.unique(data).tolist())
    unexpected = sorted(unique - expected)
    return unexpected


def _make_nnunet_id(index: int) -> str:
    """Zero-padded 3-digit case ID string."""
    return f"{index:03d}"


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert(
    fets_root: str,
    nnunet_raw: str,
    dataset_id: int = 1,
    dataset_name: str = "FeTS",
    test_subject_ids: Optional[list[str]] = None,
    verify_labels: bool = True,
    dry_run: bool = False,
    symlink: bool = False,
) -> None:
    """
    Parameters
    ----------
    fets_root       : root directory of the FeTS dataset
    nnunet_raw      : nnUNet_raw environment directory
    dataset_id      : integer dataset ID (pads to 3 digits, e.g. 1 → '001')
    dataset_name    : appended after the ID, e.g. 'FeTS' → 'Dataset001_FeTS'
    test_subject_ids: list of original subject folder names to place in imagesTs
                      instead of imagesTr (no labels copied for these)
    verify_labels   : run nibabel sanity-check on each segmentation mask
    dry_run         : print actions without touching the filesystem
    symlink         : create symlinks instead of copying (saves disk space)
    """

    fets_root  = Path(fets_root).expanduser().resolve()
    nnunet_raw = Path(nnunet_raw).expanduser().resolve()

    dataset_folder = nnunet_raw / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr = dataset_folder / "imagesTr"
    images_ts = dataset_folder / "imagesTs"
    labels_tr = dataset_folder / "labelsTr"

    test_ids_set = set(test_subject_ids or [])
    expected_label_values = set(BRATS_LABELS.values())  # {0, 1, 2, 4}

    # ------------------------------------------------------------------
    # Discover subjects
    # ------------------------------------------------------------------
    subject_dirs = _find_subject_dirs(fets_root)
    if not subject_dirs:
        print(f"[ERROR] No subject directories found under {fets_root}")
        sys.exit(1)

    print(f"Found {len(subject_dirs)} subject(s) under {fets_root}")

    # ------------------------------------------------------------------
    # Create output directories
    # ------------------------------------------------------------------
    if not dry_run:
        for d in (images_tr, images_ts, labels_tr):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Iterate subjects
    # ------------------------------------------------------------------
    training_cases: list[str] = []
    errors: list[str] = []
    label_warnings: list[str] = []

    tr_index = 1   # running counter for training cases
    ts_index = 1   # running counter for test cases

    for subject_dir in subject_dirs:
        original_id = subject_dir.name
        is_test = original_id in test_ids_set

        if is_test:
            case_id  = _make_nnunet_id(ts_index)
            ts_index += 1
            dest_img_dir = images_ts
        else:
            case_id  = _make_nnunet_id(tr_index)
            tr_index += 1
            dest_img_dir = images_tr

        case_name = f"FeTS_{case_id}"
        missing   = []

        # ---- modalities ----
        for suffix, modality_idx in MODALITY_SUFFIXES.items():
            src = _resolve_file(subject_dir, suffix)
            if src is None:
                missing.append(suffix)
                continue

            dst = dest_img_dir / f"{case_name}_{modality_idx}.nii.gz"
            _transfer(src, dst, symlink=symlink, dry_run=dry_run)

        # ---- segmentation (training only) ----
        if not is_test:
            seg_src = _resolve_file(subject_dir, SEG_SUFFIX)
            if seg_src is None:
                missing.append(SEG_SUFFIX)
            else:
                # Optional label sanity check
                if verify_labels and _NIBABEL_AVAILABLE:
                    unexpected = _check_labels(seg_src, expected_label_values)
                    if unexpected:
                        label_warnings.append(
                            f"  {original_id}: unexpected label values {unexpected}"
                        )

                seg_dst = labels_tr / f"{case_name}.nii.gz"
                _transfer(seg_src, seg_dst, symlink=symlink, dry_run=dry_run)

            training_cases.append(case_name)

        if missing:
            errors.append(f"  {original_id} (→ {case_name}): missing {missing}")
        else:
            status = "TEST " if is_test else "TRAIN"
            print(f"  [{status}] {original_id}  →  {case_name}")

    # ------------------------------------------------------------------
    # Write dataset.json
    # ------------------------------------------------------------------
    dataset_json = {**DATASET_JSON_TEMPLATE, "numTraining": len(training_cases)}
    json_path = dataset_folder / "dataset.json"

    if not dry_run:
        with open(json_path, "w") as f:
            json.dump(dataset_json, f, indent=2)
        print(f"\nWrote {json_path}")
    else:
        print(f"\n[DRY RUN] Would write dataset.json with {len(training_cases)} training cases")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Conversion complete.")
    print(f"  Training cases : {len(training_cases)}")
    print(f"  Test cases     : {ts_index - 1}")
    print(f"  Output dir     : {dataset_folder}")

    if errors:
        print(f"\n[WARNINGS] Missing files in {len(errors)} subject(s):")
        for e in errors:
            print(e)

    if label_warnings:
        print(f"\n[LABEL WARNINGS] Unexpected values in {len(label_warnings)} mask(s):")
        for w in label_warnings:
            print(w)
        print("  → Verify these manually. nnUNet will fail if labels outside")
        print("    dataset.json values are present in the masks.")

    if not _NIBABEL_AVAILABLE and verify_labels:
        print("\n[INFO] nibabel not installed — label verification skipped.")
        print("       pip install nibabel   to enable it.")

    print(f"\nNext step:")
    print(f"  nnUNetv2_plan_and_preprocess -d {dataset_id:03d} -pl nnUNetPlannerResEncM")


def _transfer(src: Path, dst: Path, symlink: bool, dry_run: bool) -> None:
    """Copy or symlink src → dst, respecting dry_run."""
    action = "SYMLINK" if symlink else "COPY"
    if dry_run:
        print(f"    [DRY RUN] {action}: {src.name} → {dst}")
        return
    if symlink:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FeTS (BraTS legacy format) dataset to nnUNet v2 raw format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "fets_root",
        help="Root directory of the FeTS dataset (contains one folder per subject).",
    )
    parser.add_argument(
        "nnunet_raw",
        help="Path to nnUNet_raw directory (value of $nnUNet_raw env var).",
    )
    parser.add_argument(
        "--dataset-id", "-d",
        type=int,
        default=1,
        help="Integer dataset ID (default: 1 → 'Dataset001_FeTS').",
    )
    parser.add_argument(
        "--dataset-name", "-n",
        default="FeTS",
        help="Dataset name suffix (default: 'FeTS').",
    )
    parser.add_argument(
        "--test-ids",
        nargs="*",
        metavar="SUBJECT_ID",
        default=None,
        help=(
            "Original subject folder name(s) to treat as held-out test cases "
            "(copied to imagesTs, no label copied). "
            "Example: --test-ids FeTS2022_00042 FeTS2022_00099"
        ),
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying files (saves disk space).",
    )
    parser.add_argument(
        "--no-verify-labels",
        action="store_true",
        help="Skip nibabel label sanity check (faster, not recommended).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without touching the filesystem.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert(
        fets_root=args.fets_root,
        nnunet_raw=args.nnunet_raw,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        test_subject_ids=args.test_ids,
        verify_labels=not args.no_verify_labels,
        dry_run=args.dry_run,
        symlink=args.symlink,
    )