import json
import yaml
import numpy as np

def compute_stats(filepath):
    gen_data = []
    val_data = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # Phân loại task dựa vào output
            if d["output"].startswith("Error:"):
                val_data.append(d)
            else:
                intent = yaml.safe_load(d["input"])
                n_vnis = len(intent["overlay"]["vnis"])
                n_spines = len(intent["underlay"]["spines"])
                n_lines = len(d["output"].strip().split("\n"))
                gen_data.append({"vnis": n_vnis, "spines": n_spines, "lines": n_lines})

    print(f"\n{'='*40}")
    print(f"File: {filepath}")
    print(f"Total samples:       {len(gen_data) + len(val_data)}")
    print(f"  Generation task:   {len(gen_data)}")
    print(f"  Validation task:   {len(val_data)}")

    vnis   = [x["vnis"]   for x in gen_data]
    spines = [x["spines"] for x in gen_data]
    lines  = [x["lines"]  for x in gen_data]

    print(f"\n[VNIs per Sample] (generation only)")
    print(f"  Min:    {min(vnis)}")
    print(f"  Max:    {max(vnis)}")
    print(f"  Mean:   {np.mean(vnis):.2f}")
    print(f"\n[CLI Output Length (lines)] (generation only)")
    print(f"  Min:    {min(lines)}")
    print(f"  Max:    {max(lines)}")
    print(f"  Mean:   {np.mean(lines):.2f}")
    print(f"  Median: {np.median(lines):.2f}")
    print(f"\n[Spine Neighbors per Sample] (generation only)")
    print(f"  Min:    {min(spines)}")
    print(f"  Max:    {max(spines)}")
    print(f"  Mean:   {np.mean(spines):.2f}")

compute_stats("dataset_vyos_train_final.jsonl")
compute_stats("dataset_vyos_val_final.jsonl")
compute_stats("dataset_vyos_test_final.jsonl")
