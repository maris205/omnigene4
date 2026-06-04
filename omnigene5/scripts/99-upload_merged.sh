#!/bin/bash
# Per-file uploader for OmniGene-4-MM-merged using classic HTTP
# (not hf-xet, which dies silently on long uploads via autodl turbo).

set -e
source /etc/network_turbo
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var with a HuggingFace write token}"
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR"

REPO="dnagpt/OmniGene-4-MM-merged"
SRC=/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-merged

# files to upload, smallest -> largest, with retries baked in via hf cli
FILES=(
  "config.json"
  "generation_config.json"
  "merge_meta.json"
  "model.safetensors.index.json"
  "processor_config.json"
  "tokenizer_config.json"
  "chat_template.jinja"
  "tokenizer.json"
  "model-00001-of-00011.safetensors"
  "model-00002-of-00011.safetensors"
  "model-00003-of-00011.safetensors"
  "model-00004-of-00011.safetensors"
  "model-00005-of-00011.safetensors"
  "model-00006-of-00011.safetensors"
  "model-00007-of-00011.safetensors"
  "model-00008-of-00011.safetensors"
  "model-00009-of-00011.safetensors"
  "model-00010-of-00011.safetensors"
  "model-00011-of-00011.safetensors"
)

ALREADY=$(python - <<EOF
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
fs = api.list_repo_files("$REPO", token=os.environ["HF_TOKEN"])
print(",".join(fs))
EOF
)
echo "Already uploaded: $ALREADY"

for f in "${FILES[@]}"; do
  if [[ ",$ALREADY," == *",$f,"* ]]; then
    echo "[SKIP $f] already on HF"
    continue
  fi
  echo "[UPLOAD $f]"
  for attempt in 1 2 3; do
    if hf upload "$REPO" "$SRC/$f" "$f" --token "$HF_TOKEN" --commit-message "Add $f" 2>&1 | tail -3; then
      echo "  attempt $attempt OK"
      break
    fi
    echo "  attempt $attempt failed, retrying..."
    sleep 10
  done
done

echo "ALL DONE"
echo "Final repo files:"
python - <<EOF
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in api.list_repo_files("$REPO", token=os.environ["HF_TOKEN"]):
    print(f"  {f}")
EOF
