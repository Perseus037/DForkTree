# DForkTree

Training-free fork-aware re-anchoring for block-parallel speculative decoding.

**Paper:** DForkTree: Training-Free Fork-Aware Re-Anchoring for Block-Parallel Speculative Decoding — arXiv link TBD.  
**Authors:** Zhicheng Zheng*, Cheng Zhang*, Chenkai Tang (* equal contribution).

DForkTree combines DFlash draft probabilities with n-gram PMI from approximately
100K target-model responses. It builds draft trees on the GPU and includes an
optimized GDN tree verifier and closed-form state commit for Qwen3.5.

## Install

Linux, Python 3.12, an NVIDIA GPU, and a CUDA-enabled PyTorch installation are required.
The reference setup uses RTX 5090, CUDA 13, and SGLang 0.5.22.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --pre -e .
```

## Run

```bash
bash scripts/download_ngram_table.sh
bash scripts/run_example.sh --model 4b
# Or: bash scripts/run_example.sh --model 9b
```

The example prints throughput, acceptance length τ, and speedup against native AR
on a few GSM8K questions. Model weights are downloaded from
[Qwen](https://huggingface.co/Qwen) and [Z Lab](https://huggingface.co/z-lab).
The shared n-gram table is hosted in this repository's Releases, not Git history.

Defaults: batch size 1, BF16, thinking disabled, B=32, block size 24, M=16,
λ_a=0.3, λ_p=0.6, and PMI clipped to [-10, 10]. The implementation retains
`W=max(B−1, M)` candidates per layer (W=31 with the defaults), without forcing
the top-1 chain into the tree. The first draft position is unchanged.

To use local files, pass `--target`, `--draft`, and `--ngram-table`.
To build a table from tokenized replies:

```bash
python scripts/build_ngram_table.py --data replies.jsonl --out data/ngram_counts.npz
```

Each JSONL row contains `resp_ids`. The table builder reuses the original counting
code; interpolated smoothing is applied by the GPU lookup at inference time.

## Code

`dforktree/` contains PMI lookup, GPU tree construction, the SGLang plugin, and
GDN verification/commit kernels. The plugin replaces the native GDN tree-verification
path. Experiment-specific baselines and analysis scripts are not included.

## Citation and license

BibTeX will be added with the arXiv release. Code: Apache-2.0; third-party notices
are in `NOTICE` and `licenses/`. Model weights and generated response text are not included.
