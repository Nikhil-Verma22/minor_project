import re
import subprocess
import sys
import os

def download_gdrive_file(file_id, out_path, html_cache):
    # Step 1: fetch page and cookie
    cookie_path = f".cache/cookie_{file_id}.txt"
    os.makedirs(".cache", exist_ok=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    cmd1 = [
        "curl.exe", "-c", cookie_path, "-s", "-L",
        f"https://drive.google.com/uc?id={file_id}&export=download",
        "-o", html_cache
    ]
    subprocess.run(cmd1, check=True)
    
    with open(html_cache, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
        
    uuid_match = re.search(r'name="uuid" value="([^"]+)"', html)
    if not uuid_match:
        print(f"Could not find uuid for {file_id}. HTML preview:")
        print(html[:500])
        return False
        
    uuid_val = uuid_match.group(1)
    print(f"Found UUID: {uuid_val}. Starting download to {out_path}...")
    
    download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t&uuid={uuid_val}"
    cmd2 = [
        "curl.exe", "-b", cookie_path, "-L",
        download_url,
        "-o", out_path
    ]
    subprocess.run(cmd2, check=True)
    
    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print(f"Downloaded {out_path} successfully. Size: {size / (1024*1024):.2f} MB")
        return True
    return False

if __name__ == "__main__":
    file_id = sys.argv[1]
    out_path = sys.argv[2]
    cache = f".cache/page_{file_id}.html"
    download_gdrive_file(file_id, out_path, cache)
