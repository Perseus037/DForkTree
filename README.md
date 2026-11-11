<div align="center">

# 🌳 DForkTree

### Accelerating Speculative Decoding with<br>Fork-Aware Block Diffusion Draft Trees

**Zhicheng Zheng<sup>*</sup> · Cheng Zhang<sup>*</sup> · Chenkai Tang**  
<sup>*</sup> Equal contribution

<a href="https://arxiv.org/abs/2611.12356"><img src="assets/badges/arxiv.svg" alt="arXiv paper"></a>
&nbsp;
<a href="https://arxiv.org/pdf/2611.12356"><img src="assets/badges/paper.svg" alt="Read the paper PDF"></a>
&nbsp;
<a href="https://github.com/Perseus037/DForkTree/releases/tag/v0.1.0"><img src="assets/badges/table.svg" alt="Download n-gram count table"></a>
&nbsp;
<a href="LICENSE"><img src="assets/badges/license.svg" alt="Apache 2.0 license"></a>

**Training-free · Fork-aware scoring · GPU tree construction**

</div>

---

## 🚀 News

- **November 9, 2026** — Our [paper](https://arxiv.org/abs/2611.12356) is available on arXiv.
- The core implementation and [shared n-gram count table](https://github.com/Perseus037/DForkTree/releases/tag/v0.1.0) are available in this repository.

## 💡 Overview

A draft tree gains over a chain through branches that leave the drafter’s preferred continuation. Yet the original per-position probabilities become less reliable after a fork.

**DForkTree re-anchors each branch on its own tokens**, using n-gram statistics from approximately **100K target-model responses**:

- **🌿 Fork-aware scoring.** Add PMI evidence from the branch’s last two tokens, with a larger weight after leaving the top-1 path. Retain the probability mass outside the candidate list.
- **⚡ GPU tree construction.** Batch the n-gram lookups and build the tree under the corrected scores inside a CUDA graph, without forcing the top-1 chain into the tree.
- **🧩 Efficient verification.** Integrate with SGLang and use an optimized GDN tree verifier with closed-form state commit for Qwen3.5.

No additional training or extra drafter forward pass is required.

> **Paper highlights:** up to **4.75× average speedup over autoregressive decoding**, with gains of up to **41.5% over DFlash** and **15.1% over DDTree** in the evaluated settings. See the [paper](https://arxiv.org/abs/2611.12356) for the full comparisons.

## 📦 Core Code & Data

| Component | Source |
| :--- | :--- |
| PMI lookup and smoothing | [`ngram_gpu.py`](dforktree/ngram_gpu.py) |
| GPU tree construction | [`fc1_builder.py`](dforktree/fc1_builder.py) · [`tree_ops.py`](dforktree/tree_ops.py) |
| GDN verification and state commit | [`tree_kernels.py`](dforktree/tree_kernels.py) · [`fc1_fused.py`](dforktree/fc1_fused.py) |
| SGLang integration | [`fc1_plugin.py`](dforktree/fc1_plugin.py) · [`fc1_dflash_worker.py`](dforktree/fc1_dflash_worker.py) |
| Shared n-gram table for 4B and 9B | [**Download `ngram_counts.npz`**](https://github.com/Perseus037/DForkTree/releases/download/v0.1.0/ngram_counts.npz) |

## ⚙️ Configuration

**Defaults:** B=32 · block size 24 · M=16 · λ_a=0.3 · λ_p=0.6 · PMI clipping [−10, 10].

The implementation uses `W=max(B−1, M)` (W=31 by default). The first draft position is unchanged.

**Reference environment:** Linux · Python 3.12 · CUDA 13 · SGLang 0.5.22 · RTX 6000PRO.

## 📖 Citation

```bibtex
@article{zheng2026dforktree,
  title   = {DForkTree: Accelerating Speculative Decoding with Fork-Aware Block Diffusion Draft Trees},
  author  = {Zheng, Zhicheng and Zhang, Cheng and Tang, Chenkai},
  journal = {arXiv preprint arXiv:2611.12356},
  year    = {2026},
  url     = {https://arxiv.org/abs/2611.12356}
}
```

## 📄 License

[Apache-2.0](LICENSE). See [`NOTICE`](NOTICE) for upstream attribution.

