#!/usr/bin/env python
"""
Robust per-file uploader for OmniGene-4-MM-LoRA.
Uploads each file separately so a failure mid-way does not lose previous files.
"""
import os
import time
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

REPO = "dnagpt/OmniGene-4-MM-LoRA"
SRC  = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage3v3"

FILES = [
    "lora_weights.pt",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "meta.json",
    "embedding_weights.pt",      # largest, do last
]

def already_uploaded(api, repo, fname):
    try:
        files = api.list_repo_files(repo)
        return fname in files
    except Exception:
        return False

def upload_one(api, repo, src_dir, fname, retries=4):
    full_path = os.path.join(src_dir, fname)
    if not os.path.exists(full_path):
        print(f"  SKIP (missing): {fname}")
        return
    if already_uploaded(api, repo, fname):
        print(f"  SKIP (already on HF): {fname}")
        return

    size_mb = os.path.getsize(full_path) / 1e6
    for attempt in range(retries):
        try:
            print(f"  Uploading {fname} ({size_mb:.1f} MB), attempt {attempt+1}/{retries}...", flush=True)
            t0 = time.time()
            api.upload_file(
                path_or_fileobj=full_path,
                path_in_repo=fname,
                repo_id=repo,
                commit_message=f"Add {fname}",
            )
            dt = time.time() - t0
            print(f"    OK ({size_mb/dt:.2f} MB/s, {dt:.1f}s)", flush=True)
            return
        except (HfHubHTTPError, OSError, ConnectionError) as e:
            print(f"    FAIL: {type(e).__name__}: {str(e)[:200]}", flush=True)
            if attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"    sleeping {wait}s before retry", flush=True)
                time.sleep(wait)
            else:
                print(f"    GIVING UP on {fname}", flush=True)
                raise

def main():
    api = HfApi(token=os.environ["HF_TOKEN"])
    for f in FILES:
        upload_one(api, REPO, SRC, f)
    print("\nFinal HF state:")
    for f in api.list_repo_files(REPO):
        print(f"  {f}")

if __name__ == "__main__":
    main()
