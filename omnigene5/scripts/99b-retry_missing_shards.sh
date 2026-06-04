#!/bin/bash
# Aggressively retry the 3 still-missing shards.
set -e
source /etc/network_turbo
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var with a HuggingFace write token}"
export TMPDIR=/root/autodl-tmp/tmp

REPO="dnagpt/OmniGene-4-MM-merged"
SRC=/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-merged

for i in 02 07 10; do
  f="model-000${i}-of-00011.safetensors"
  ON_HF=$(python - <<EOF
import os
from huggingface_hub import HfApi
fs = HfApi(token=os.environ["HF_TOKEN"]).list_repo_files("$REPO")
print("YES" if "$f" in fs else "NO")
EOF
)
  if [ "$ON_HF" = "YES" ]; then
    echo "[SKIP $f] already on HF"
    continue
  fi
  echo "[UPLOAD $f]"
  for attempt in $(seq 1 8); do
    set +e
    hf upload "$REPO" "$SRC/$f" "$f" --token "$HF_TOKEN" --commit-message "Add $f"
    rc=$?
    set -e
    sleep 5
    ON_HF=$(python - <<EOF
import os
from huggingface_hub import HfApi
fs = HfApi(token=os.environ["HF_TOKEN"]).list_repo_files("$REPO")
print("YES" if "$f" in fs else "NO")
EOF
)
    if [ "$ON_HF" = "YES" ]; then
      echo "  attempt $attempt OK (rc=$rc)"
      break
    fi
    echo "  attempt $attempt: rc=$rc, on-HF=$ON_HF; retrying..."
    sleep $((attempt * 15))
  done
done

echo "FINAL STATE:"
python - <<EOF
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
fs = sorted(api.list_repo_files("$REPO"))
shards = [f for f in fs if "safetensors" in f and "index" not in f]
print(f"have {len(shards)}/11 shards")
for f in shards: print(f"  {f}")
EOF
