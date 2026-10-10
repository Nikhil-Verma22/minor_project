import os
import hashlib
import json
import sys

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def inventory_directory(root_dir, output_files):
    inventory = {}
    print(f"Scanning {root_dir}...")
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Sort for deterministic processing
        dirnames.sort()
        filenames.sort()
        for f in filenames:
            full_path = os.path.join(dirpath, f)
            rel_path = os.path.relpath(full_path, root_dir).replace("\\", "/")
            file_sha = compute_sha256(full_path)
            file_size = os.path.getsize(full_path)
            inventory[rel_path] = {
                "sha256": file_sha,
                "size_bytes": file_size
            }
            file_count += 1
            if file_count % 100 == 0:
                print(f"Hashed {file_count} files...")
                
    print(f"Completed hashing {file_count} files.")
    for out_path in output_files:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(inventory, out_f, indent=2)
        print(f"Saved inventory to {out_path}")

if __name__ == "__main__":
    eval_dir = "d:/Minor/evaluation_sets"
    out_paths = [
        "d:/Minor/.cache/REACT/eval_sets_sha256_pre.json",
        "d:/Minor/results/REACT/eval_sets_sha256_pre.json"
    ]
    inventory_directory(eval_dir, out_paths)
