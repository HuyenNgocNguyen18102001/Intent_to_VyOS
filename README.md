# Intent-to-VyOS: Benchmarking Domain-Adaptive Instruction Tuning for EVPN/VXLAN Configuration Synthesis

> **ISBCom 2026** | Ngoc Huyen Nguyen, Duc Dat Pham, Kha Tu Huynh, Tan Duy Le*, Nguyen Tan Viet Tuyen
>
> *Corresponding author: ldtan@hcmiu.edu.vn

---

## Overview

**Intent-to-VyOS** is a benchmark dataset and evaluation framework for assessing domain-adaptive instruction-tuned large language models on **EVPN/VXLAN configuration synthesis for VyOS**.

The benchmark maps high-level YAML network intents to full VyOS CLI configurations and evaluates models using a multi-level protocol combining **Exact Match**, **Token F1**, **Command F1**, and **Truncation Rate**.

Our key finding: **Qwen2.5-3B outperforms all 7B-scale models** — demonstrating that output stability and domain specialization, not raw parameter count, drive performance in structured configuration generation.

---

## Repository Structure

```
Intent_to_VyOS/
├── dataset/
│   ├── dataset_vyos_train_final.jsonl   # 3,000 training samples
│   ├── dataset_vyos_val_final.jsonl     # 300 validation samples
│   ├── dataset_vyos_test_final.jsonl    # 300 test samples
│   └── stat_data.PY                     # Dataset statistics script
└── examples/
    └── leaf1_vxlan.yaml                 # Example YAML intent document
```

---

## Dataset

### Tasks

The benchmark defines two complementary tasks:

**Generation Task** — Given a natural language instruction and a structured YAML network intent, generate the complete VyOS CLI configuration.

**Validation Task** — Given a malformed or semantically inconsistent YAML intent, produce a structured error message identifying the violation class and location.

### Input Format

Each sample is a JSONL record with three fields:

```json
{
  "instruction": "Build a VXLAN overlay configuration for VyOS using these parameters.",
  "input": "<YAML network intent>",
  "output": "<VyOS CLI set commands or structured error message>"
}
```

The YAML intent encodes two blocks:

```yaml
metadata:
  device: <leaf-name>
  role: leaf
underlay:
  bgp_asn: <private-ASN>        # drawn from 65000–65535
  router_id: <loopback-IP>
  spines:
    - name: <spine-name>
      neighbor_ip: <point-to-point-IP>
      remote_asn: <spine-ASN>
overlay:
  vnis:
    - id: <VNI-id>
      vlan: <VLAN-id>
      vrf: <VRF-name>
      gateway: <anycast-GW-prefix>
      description: <tenant-description>
```

The target CLI output covers: system hostname, BGP underlay peering with L2VPN EVPN address-family activation per spine neighbor, global VXLAN interface with `advertise-all-vni`, and per-VNI blocks (VXLAN VNI binding, bridge interface membership, SVI gateway, `advertise-svi-ip`).

### Statistics

| Statistic | Train | Val | Test |
|---|---|---|---|
| Total samples | 3,000 | 300 | 300 |
| Generation task | 2,699 | 269 | 273 |
| Validation task | 301 | 31 | 27 |
| VNIs per sample (min/max/mean) | 1/3/1.95 | 1/3/1.97 | 1/3/1.99 |
| CLI lines (min/max/mean) | 13/21/16.79 | 13/21/16.87 | 17/25/20.96 |
| Spine neighbors (fixed count) | 2 | 2 | **4** |
| Unique instruction variants | 10 | 10 | 11 |

> **Note on distributional shift:** Training uses 2 spine neighbors per leaf; test uses 4. This deliberate out-of-distribution probe evaluates structural generalization to unseen topology configurations.

### Data Split and Leakage Prevention

- **Topology-based split**: all instances from the same template family and spine-count are assigned exclusively to one split.
- **Hash-based deduplication**: SHA-256 over `(instruction, input, output)` — any duplicate across splits is removed.
- **Triple non-overlap**: no `(instruction, input)` pair and no `(input, output)` pair from training appears in validation or test.

---

## Models Evaluated

| Model | Parameters | Family |
|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | Qwen2.5 |
| Qwen2.5-3B-Instruct | 3B | Qwen2.5 |
| Qwen2.5-7B-Instruct | 7B | Qwen2.5 |
| Gemma-2-2B-IT | 2B | Gemma-2 |
| Mistral-7B-Instruct-v0.3 | 7B | Mistral |

All models are fine-tuned with **QLoRA** (4-bit NF4, LoRA rank 16, alpha 32, all linear layers) using the following ChatML prompt template:

```
<|im_start|>system
You are a VyOS network configuration assistant.<|im_end|>
<|im_start|>user
{instruction}

{yaml_intent}<|im_end|>
<|im_start|>assistant
{target_cli_output}<|im_end|>
```

**Training hyperparameters:** learning rate `2e-4`, Paged AdamW (8-bit), 3 epochs, effective batch size 16 (4×4), max sequence length 2,048 tokens.

**Inference:** greedy decoding (`do_sample=False`), `max_new_tokens=1024`, normalized stop tokens across model families.

---

## Evaluation Metrics

| Level | Metrics |
|---|---|
| Text-level | Exact Match (EM), ROUGE-L, BLEU |
| Token-level | Token Precision, Token Recall, Token F1 |
| **Command-level** | **Command Precision, Command Recall, Command F1** ← primary |
| Structural | Truncation Rate |

**Command F1** is the primary metric. Each `set` directive is treated as an atomic unit; outputs are normalized (whitespace collapsed, leading/trailing stripped) then split by line into command sets for comparison.

**Truncation Rate** measures the proportion of outputs that exhaust `max_new_tokens` without generating a natural EOS token — a critical diagnostic for long-form structured generation.

---

## Results

| Model | EM | Token F1 | **Cmd F1** | ROUGE-L | BLEU | Trunc. |
|---|---|---|---|---|---|---|
| **Qwen2.5-3B** | **0.920** | 0.996 | **0.991** | 0.989 | 0.981 | **0.000** |
| Qwen2.5-1.5B | 0.597 | 0.979 | 0.979 | 0.968 | 0.942 | 0.003 |
| Qwen2.5-7B | 0.090 | 0.711 | 0.385 | 0.772 | 0.620 | 0.187 |
| Mistral-7B | 0.000 | 0.583 | 0.321 | 0.520 | 0.259 | 0.290 |
| Gemma-2-2B | 0.000 | 0.452 | 0.260 | 0.366 | 0.138 | 0.357 |

**Key finding:** Qwen2.5-3B achieves the highest Command F1 (0.991) with zero truncation, outperforming all 7B-scale models. Model size alone is not a reliable predictor of performance in domain-specific structured generation.

---

## Citation

If you use this dataset or evaluation framework, please cite:

```bibtex
@inproceedings{nguyen2026intentvyos,
  title     = {Intent-to-VyOS: Benchmarking Domain-Adaptive Instruction Tuning for EVPN/VXLAN Configuration Synthesis},
  author    = {Nguyen, Ngoc Huyen and Pham, Duc Dat and Huynh, Kha Tu and Le, Tan Duy and Tuyen, Nguyen Tan Viet},
  booktitle = {ISBCom 2026},
  year      = {2026}
}
```

---

## Acknowledgments

The authors thank **AIoT Lab VN** for support throughout this project. This research is also supported by the central Interdisciplinary Laboratory in Electronics and Information Technology (AI and Cooperation Robot), International University – VNU-HCM.
