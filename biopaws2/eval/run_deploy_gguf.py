"""Local-deployment validation: OmniGene-4 v3 as Q4_K_M GGUF on a single 24GB GPU.

Produces the deployment table the reviewer asked for:
  - accuracy on BioPAWS-2 tasks (quantized GGUF) vs the unquantized v3 reference,
  - peak VRAM footprint,
  - per-query latency and throughput.

Runs the public GGUF (dnagpt/OmniGene-4-SFT-v3-GGUF, Q4_K_M) via llama-cpp-python with
full GPU offload, on a slice of BioPAWS-2 test items, using the same Alpaca prompt + scorer
as the main eval harness. This substantiates the "locally deployable" claim quantitatively.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.score import score_task  # noqa: E402

GGUF = "/root/autodl-tmp/dnagpt/gguf_v3/OmniGene-4-SFT-v3-Q4_K_M.gguf"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def alpaca(instr):
    return f"### Instruction:\n{instr}\n\n### Answer:\n"


def load_test(task, limit):
    rows = []
    with open(os.path.join(DATA, f"{task}.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("split") == "test":
                    rows.append(r)
    return rows[:limit]


def vram_mb():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return int(out.decode().strip().split("\n")[0])
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+",
                    default=["protein_homology_std", "protein_homology_remote",
                             "lg_promoter_detection", "lg_signal_peptide"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)  # -1 = all on GPU
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    from llama_cpp import Llama

    base_vram = vram_mb()
    t0 = time.time()
    llm = Llama(model_path=GGUF, n_gpu_layers=a.n_gpu_layers, n_ctx=a.ctx,
                verbose=False, logits_all=False)
    load_s = time.time() - t0
    loaded_vram = vram_mb()
    print(f"[deploy] loaded in {load_s:.1f}s | VRAM {base_vram}->{loaded_vram} MB "
          f"(+{loaded_vram-base_vram} MB)", flush=True)

    per_task = {}
    all_latencies = []
    for task in a.tasks:
        rows = load_test(task, a.limit)
        if not rows:
            continue
        preds = {}
        lat = []
        for r in rows:
            prompt = alpaca(r["messages"][0]["content"])
            t = time.time()
            out = llm(prompt, max_tokens=a.max_tokens, temperature=0.0, echo=False)
            lat.append(time.time() - t)
            preds[r["id"]] = out["choices"][0]["text"].strip()
        res = score_task(os.path.join(DATA, f"{task}.jsonl"), preds)
        import statistics
        med = statistics.median(lat)
        per_task[task] = {"score": res["score"], "metric": res["metric"], "n": res["n"],
                          "median_latency_s": round(med, 3),
                          "throughput_qps": round(1.0 / med, 2) if med else 0}
        all_latencies += lat
        print(f"[deploy] {task}: {res['score']:.3f} ({res['metric']}, n={res['n']}) "
              f"| median {med:.3f}s/query", flush=True)

    import statistics
    peak_vram = vram_mb()
    summary = {
        "model": "OmniGene-4-SFT-v3", "quantization": "Q4_K_M GGUF",
        "gpu": "single 24GB-class (measured on available card)",
        "load_time_s": round(load_s, 1),
        "model_vram_mb": loaded_vram - base_vram,
        "peak_vram_mb": peak_vram,
        "median_latency_s": round(statistics.median(all_latencies), 3) if all_latencies else 0,
        "throughput_qps": round(1.0 / statistics.median(all_latencies), 2) if all_latencies else 0,
        "per_task": per_task,
    }
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(summary, open(os.path.join(a.out_dir, "deploy_gguf_v3.json"), "w"), indent=2)
    print(f"[deploy] SUMMARY {json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
