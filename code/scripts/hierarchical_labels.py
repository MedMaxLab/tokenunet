import json

path1 = "/data/tshimanga/nnUNet_raw/Dataset001_FeTS/dataset.json"
path2 = "/data/tshimanga/nnUNet_preprocessed/Dataset001_FeTS/dataset.json"

for path in [path1, path2]:
    with open(path) as f:
        d = json.load(f)

    d["labels"] = {
        "background": 0,
        "NCR":        1,
        "ED":         2,
        "ET":         3,
    }

    # Regions are defined as unions of the non-overlapping labels above.
    # nnUNet will derive these at evaluation time — no mask changes needed.
    d["regions_class_order"] = [1, 2, 3]   # order in which regions are predicted
    d["regions"] = {
        "ET": [3],          # Enhancing Tumour      — label 3 only
        "TC": [1, 3],       # Tumour Core           — NCR + ET
        "WT": [1, 2, 3],    # Whole Tumour          — NCR + ED + ET
    }

    with open(path, "w") as f:
        json.dump(d, f, indent=2)

print("done")