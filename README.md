# DForkTree

Core code for **DForkTree: Training-Free Fork-Aware Re-Anchoring for Block-Parallel Speculative Decoding**.

Zhicheng Zheng*, Cheng Zhang*, Chenkai Tang (* equal contribution). ArXiv link: TBD.

DForkTree combines DFlash draft probabilities with n-gram PMI from approximately 100K target-model responses. This repository contains PMI lookup, GPU tree construction, the GDN tree verifier with closed-form state commit, and SGLang integration.

Defaults: B=32, block size 24, M=16, λ_a=0.3, λ_p=0.6, PMI clipping [-10, 10]. The implementation uses `W=max(B−1, M)` (W=31 by default), without forcing the top-1 chain into the tree.

Environment: Linux, Python 3.12, CUDA 13, SGLang 0.5.22; reference GPU: RTX 5090.

N-gram count table: [download from Releases](https://github.com/Perseus037/DForkTree/releases/download/v0.1.0/ngram_counts.npz).

Model weights, generated replies, experiment scripts and baselines are not included.

Apache-2.0. See `NOTICE` for upstream attribution.

