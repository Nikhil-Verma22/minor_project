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

def verify_and_save_inventory():
    eval_dir = "d:/Minor/evaluation_sets"
    baseline_path = "d:/Minor/results/REACT/eval_sets_sha256_post.json"
    post_output_path = "d:/Minor/results/FastContextualSGG/eval_sets_sha256_post.json"

    print(f"Loading baseline inventory from {baseline_path}...")
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    print(f"Baseline contains {len(baseline)} files.")

    current_inventory = {}
    print(f"Scanning {eval_dir} to verify SHA-256 integrity...")
    file_count = 0
    mismatches = []

    for dirpath, dirnames, filenames in os.walk(eval_dir):
        dirnames.sort()
        filenames.sort()
        for f in filenames:
            full_path = os.path.join(dirpath, f)
            rel_path = os.path.relpath(full_path, eval_dir).replace("\\", "/")
            file_sha = compute_sha256(full_path)
            file_size = os.path.getsize(full_path)

            current_inventory[rel_path] = {
                "sha256": file_sha,
                "size_bytes": file_size
            }

            if rel_path not in baseline:
                mismatches.append(f"NEW FILE ADDED: {rel_path}")
            elif baseline[rel_path]["sha256"] != file_sha:
                mismatches.append(f"CONTENT MODIFIED: {rel_path} (orig: {baseline[rel_path]['sha256']}, now: {file_sha})")
            elif baseline[rel_path]["size_bytes"] != file_size:
                mismatches.append(f"SIZE CHANGED: {rel_path}")

            file_count += 1
            if file_count % 200 == 0:
                print(f"Verified {file_count} files...")

    for rel_path in baseline:
        if rel_path not in current_inventory:
            mismatches.append(f"FILE DELETED: {rel_path}")

    print(f"Finished scanning {file_count} files.")

    if mismatches:
        print(f"ERROR: {len(mismatches)} INTEGRITY VIOLATIONS FOUND:")
        for m in mismatches:
            print("  -", m)
        sys.exit(1)
    else:
        print(f"SUCCESS: All {file_count} files are 100% BIT-FOR-BIT IDENTICAL to the baseline!")

    os.makedirs(os.path.dirname(post_output_path), exist_ok=True)
    with open(post_output_path, "w", encoding="utf-8") as f:
        json.dump(current_inventory, f, indent=2)
    print(f"Saved post-run verification inventory to {post_output_path}")

if __name__ == "__main__":
    verify_and_save_inventory()
