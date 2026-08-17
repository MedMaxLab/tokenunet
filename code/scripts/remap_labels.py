# remap_labels.py
import nibabel as nib
import numpy as np
from pathlib import Path

REMAP = {4: 3}   # ET: 4 → 3

labels_dir = Path("/data/tshimanga/nnUNet_raw/Dataset001_FeTS/labelsTr")

for mask_path in sorted(labels_dir.glob("*.nii.gz")):
    img  = nib.load(mask_path)
    data = np.asarray(img.dataobj, dtype=np.uint8)

    remapped = data.copy()
    for old, new in REMAP.items():
        remapped[data == old] = new

    nib.save(nib.Nifti1Image(remapped, img.affine, img.header), mask_path)
    print(f"Remapped {mask_path.name}  unique={np.unique(remapped).tolist()}")