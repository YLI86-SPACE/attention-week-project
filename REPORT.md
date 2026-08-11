# Week 1-3 Review Project Report

## Abstract

This repository implements the required core code for SmallCNN, Patch-LSTM, MiniViT-Full, MiniViT-Window, custom multi-head self-attention, 1D RoPE, and 2D Axial RoPE. Local verification was run on a CPU-only Windows environment, so long 20-epoch training results are not yet filled in. The lite Tiny ImageNet configuration was generated deterministically and fast numerical experiments were completed.

## Project Facts

- Data configuration and class count: lite, 100 classes
- Split manifest SHA-256: `1cb97ea048322c7e7625cd43b895f9425ae60b6999b19eb3ab063f206ae76bce`
- Seeds: 42, 43 for Full/Window paired runs
- Hardware: Windows CPU environment
- PyTorch/CUDA version: PyTorch 2.13.0+cpu, CUDA unavailable locally
- Precision: FP32 for tests and quick experiments
- Effective batch size: 64 for training configs

## Questions

### RQ1: Why Divide Attention Scores by sqrt(d_k)?

Let `q = (q_1, ..., q_dk)`, `k = (k_1, ..., k_dk)`, and `z = q^T k = sum_r q_r k_r`. Under the random assumptions `E[q_r]=E[k_r]=0`, `Var(q_r)=Var(k_r)=1`, independence between `q` and `k`, and independence across dimensions:

```text
E[z] = E[sum_r q_r k_r]
     = sum_r E[q_r k_r]
     = sum_r E[q_r] E[k_r]
     = 0
```

For each dimension, `Var(q_r k_r) = E[q_r^2 k_r^2] - E[q_r k_r]^2 = E[q_r^2]E[k_r^2] = 1`. Cross terms vanish by independence, so:

```text
Var(z) = Var(sum_r q_r k_r) = sum_r Var(q_r k_r) = d_k
Std(z) = sqrt(d_k)
```

For `z_scaled = q^T k / sqrt(d_k)`:

```text
E[z_scaled] = 0
Var(z_scaled) = Var(z) / d_k = 1
```

This is a design argument, not a guarantee for every training state. Real learned Q/K can be correlated and need not have unit variance.

For `p = softmax(z)`, the row-wise Jacobian is:

```text
J = diag(p) - p p^T
partial p_i / partial z_j = p_i (delta_ij - p_j)
```

When the softmax distribution approaches one-hot, `J` approaches zero. However, `||J||_F` is not globally guaranteed to decrease monotonically with every possible score scaling; the experiment directly measures the configured `d_k` range. This does not imply the whole model's gradients always vanish, especially because final gradients depend on the loss, V, and downstream layers.

Local numeric result from `outputs/experiments/scaling.json`: finite-difference sensitivity check used `epsilon=1e-5`; finite difference was `0.0216101503`, while `||Ju||_2` was `0.0216101580`, absolute error `7.72e-09`. The generated figure is `outputs/figures/attention_scaling.png`.

### RQ2: What Does Full Attention Cost, and How Can It Be Improved?

With batch size `B`, token count `N`, embedding dimension `D`, heads `H`, head dimension `d_h=D/H`, window side `w`, window tokens `M=w^2`, and MLP ratio `r`:

```text
Full Attention:
QKV and output projection: Theta(B N D^2)
QK^T and AV:              Theta(B N^2 D)
score elements:            B H N^2
full block:                Theta(B N D^2 + B N^2 D + r B N D^2)

Window Attention:
number of windows:         N / M
QK^T and AV:               Theta(B N M D)
score elements:            B H N M
full block:                Theta(B N D^2 + B N M D + r B N D^2)
```

For naive explicit-score implementations, main forward activations are `Theta(BND + BHN^2)` for Full Attention and `Theta(BND + BHNM)` for Window Attention. Real training peak memory is larger because backward activations, MLP activations, parameters, gradients, and optimizer state also matter.

Improvement families:

| Family | Mechanism | Exact? | Complexity effect | Main use | Possible loss |
|---|---|---|---|---|---|
| Local/window | attend only inside neighborhoods | sparse exact over allowed edges | `N^2` to `NM` | train/infer | misses cross-window mixing |
| Sparse patterns | selected global/local/random edges | sparse exact over pattern | depends on pattern | long context | omitted edges lose information |
| Linear/kernel | kernelize softmax-like attention | approximate | often near linear | long context | approximation error |
| FlashAttention | IO-aware tiling of exact attention | exact | arithmetic remains quadratic | train/infer memory and speed | no sparsity, still `N^2D` compute |
| MQA/GQA | share K/V across heads or groups | exact for changed architecture | reduces KV cache and K/V bandwidth | autoregressive inference | less per-head K/V diversity |
| Low-rank | factorize attention or projections | approximate | can reduce `N^2` terms | long context | rank bottleneck |
| Latent bottleneck | tokens attend through small latent set | approximate/architectural | `N L` with latent count `L` | multimodal/long input | bottleneck compression |
| Token pruning/merging | reduce number of tokens dynamically | approximate | smaller effective `N` | train/infer | removed/merged details |
| Factorized spatiotemporal | separate spatial and temporal attention | exact for factorized graph | avoids full all-pair video attention | video | restricted joint interactions |

Local CPU benchmark from `outputs/experiments/benchmark.json`: at `N=1024`, Full forward median was about `19.64 ms`, Window forward median about `1.57 ms`; at `N=64`, Full and Window were similar because overhead and small matrices dominate. This result is specific to CPU PyTorch 2.13.0, FP32, `B=4`, `D=128`, `H=4`, and this implementation.

### RQ3: What Dependency Graphs Do CNN, LSTM, and Attention Allow?

Structural evidence, algebraic tests, numerical perturbation, and task metrics answer different questions.

- SmallCNN: `SmallCNN` uses three Conv-ReLU-MaxPool blocks. Local receptive fields expand with convolution kernel, padding, pooling stride, and depth. Position comes from the 2D grid, local kernels, and boundary conditions.
- Patch-LSTM: `PatchLSTM` cuts the image into 256 row-major 4x4 patches. A unidirectional LSTM output at time `t` can depend only on tokens `0...t`; the final hidden state is order-sensitive.
- Full Attention: `MultiHeadSelfAttention` creates a direct path between every token pair in one layer. Without position-dependent operations, it is token permutation equivariant, and mean pooling is invariant.
- Window Attention: `WindowMultiHeadSelfAttention` restricts attention to 4x4 windows. Without shift or merge, different windows do not exchange token information before global pooling summarizes separate window features.

Algebraic pytest coverage confirms NoPos full-attention equivariance, RoPE token-coordinate pair permutation equivariance, and window-unit swap equivariance under dropout-free FP32 conditions. These tests do not prove task accuracy or causal importance.

### RQ4: How Does RoPE Put Position Into Attention?

For an even-dimensional vector, pair adjacent channels. Pair `i` uses frequency:

```text
theta_i = base^(-2i/d), base = 10000
```

The 2D rotation at position `m` is:

```text
R(m, theta_i) = [[cos(m theta_i), -sin(m theta_i)],
                 [sin(m theta_i),  cos(m theta_i)]]
```

For a channel pair `(a,b)`:

```text
a' = a cos(m theta_i) - b sin(m theta_i)
b' = a sin(m theta_i) + b cos(m theta_i)
```

Norm preservation follows because rotation matrices are orthogonal:

```text
||R_m x||_2^2 = x^T R_m^T R_m x = x^T x
```

Using orthogonality and the rotation group property:

```text
(R_m q)^T (R_n k)
= q^T R_m^T R_n k
= q^T R_(n-m) k
```

Thus RoPE makes the Q/K dot product explicitly depend on relative displacement `n-m`, while the score still also depends on content `q` and `k`.

In this project, 2D Axial RoPE splits each head of dimension 32 into row and column subspaces of 16 dimensions each. Row frequencies use `d=16`; column frequencies also use `d=16`. The attention score is:

```text
score((r1,c1),(r2,c2))
= [q_row^T R_(r2-r1) k_row + q_col^T R_(c2-c1) k_col] / sqrt(d_h)
```

The divisor remains the full head dimension `d_h=32`. Flattening a 16x16 patch grid into positions `0...255` would collapse 2D adjacency into a raster order, making vertically adjacent patches appear 16 steps apart while horizontally adjacent patches appear 1 step apart.

| Property | NoPos | Learned Absolute | RoPE |
|---|---|---|---|
| Added where | nowhere | token representation | Q/K rotation |
| Learned per-position params | no | yes | no |
| Explicit relative score structure | no | no | yes |
| Unseen positions generated directly | no position code | no fixed table extrapolation | formula can generate angles |
| Token permutation behavior | equivariant/invariant after mean | content-slot mismatch changes output | equivariant if token-coordinate pairs move together |
| 2D image handling | no coordinates | one embedding per slot | row/column axial rotation |

RoPE limitations include base/frequency choices, even-dimension constraints, phase behavior at long positions, and the distinction between formulaic extrapolation and learned resolution or length generalization.

## Hypotheses

1. Unscaled score standard deviation will grow approximately like `sqrt(d_k)` under i.i.d. unit-variance Q/K.
2. Scaled score standard deviation will stay close to 1 under the same random assumptions.
3. Window attention will have fewer score elements than full attention, but actual latency at `N=256` may depend on implementation overhead.
4. Fixed W-MSA has no cross-window token path before pooling; alternating SW-MSA would expand the theoretical reachable set.
5. Before training, I predict AbsPos >= RoPE2D >= NoPos for validation accuracy in this small fixed-grid setup, because learned slot embeddings can directly specialize to the training resolution, while RoPE2D supplies relative structure and NoPos lacks explicit position.

## Method

The lite split uses seed 42 to sample 100 WNIDs from sorted `wnids.txt`, then sorts the sampled WNIDs. Each train class is split by sorting `sha256("split-v1|42|" + relative_path)`, using 450 images for train and 50 for validation. Official Tiny ImageNet validation images become the held-out test set for the selected classes.

Images return normalized input, class id, and raw pre-normalization RGB mean. Training uses random horizontal flip only, then ToTensor and ImageNet normalization. Validation/test are deterministic.

Core model classes/functions include `SmallCNN`, `PatchLSTM`, `MultiHeadSelfAttention`, `attention_core`, `apply_rope_1d`, `apply_rope_2d`, `MiniViT`, `window_partition`, `window_reverse`, and `WindowMultiHeadSelfAttention`.

## Experiments

### Attention Scaling

Hypothesis: unscaled scores grow with `sqrt(d_k)`; scaled scores stay around unit standard deviation.
Compared groups: scaled vs unscaled random attention scores.
Changed factor: division by `sqrt(d_k)`.
Controlled factors: i.i.d. standard normal Q/K, sequence length 64, seed 42, 100 trials.
Measured metrics: score mean/std, normalized entropy, Jacobian Frobenius sensitivity.
Observed result: stored in `outputs/experiments/scaling.json` and `outputs/figures/attention_scaling.png`.
Whether the hypothesis is supported: yes for the score-scale prediction in the generated data.
Conclusion tied to evidence: scaling stabilizes score variance under the stated random assumptions.
Confounders and limitations: this does not describe all learned Q/K distributions or all model gradients.

### Full/Window Benchmark

Hypothesis: Window has fewer score elements, but runtime depends on implementation and N.
Compared groups: Full MHA vs W-MSA.
Changed factor: attention mixing range.
Controlled factors: B=4, D=128, H=4, FP32, same projection weights, CPU.
Measured metrics: forward and forward+backward latency, theoretical score bytes.
Observed result: see `outputs/experiments/benchmark.json`.
Whether the hypothesis is supported: yes; at small N overhead matters, at N=1024 Window is much faster locally.
Conclusion tied to evidence: theoretical score savings become visible at larger N in this CPU implementation.
Confounders and limitations: no CUDA memory result locally; CPU behavior does not generalize to all kernels.

## Results

### Implementation Checks

`outputs/experiments/implementation_checks.json` was generated automatically. It records the expected shapes for CNN, LSTM, MiniViT-Full, and MiniViT-Window.

### Tests

Local command: `pytest tests -q`.
Result: `14 passed`.

### Primary Classification Table

Long training not completed locally because only CPU PyTorch is available. Fill this table from `outputs/runs/<run_id>/final_metrics.json` after CUDA/Colab training.

| Model | Params | Best Val Acc | Full Test N | Full Test Acc | Train Time/Epoch |
|---|---:|---:|---:|---:|---:|
| SmallCNN | TODO | TODO | TODO | TODO | TODO |
| Patch-LSTM | TODO | TODO | TODO | TODO | TODO |
| MiniViT-Full | TODO | TODO | TODO | TODO | TODO |
| MiniViT-Window | TODO | TODO | TODO | TODO | TODO |

### CLIP Zero-Shot Reference

Local HuggingFace model download failed due a remote connection reset. The script `experiments/clip_zero_shot.py` should be run in Colab or another environment with HuggingFace access.

## Discussion

The code-level and algebraic tests support implementation properties, not broad claims about model quality. Accuracy tables, once filled, should be described as results under this data split, model size, optimizer, training budget, and seed protocol only.

## Limitations

The local machine has CPU-only PyTorch, so full 20-epoch training and CLIP zero-shot were not completed locally in this pass. Position ablation is single-seed, the grid is fixed at 16x16, and benchmark results are hardware/kernel dependent.

## Conclusion

The repository now contains the core hand-written implementations, deterministic data split pipeline, tests, and fast numerical experiments needed for the report. The remaining high-cost work is the CUDA/Colab execution of the 9-run training matrix and CLIP zero-shot evaluation.

## References

- Vaswani et al., Attention Is All You Need. Used for scaled dot-product attention and multi-head attention.
- Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding. Used for RoPE rotation and relative-position dot-product identity.
- Liu et al., Swin Transformer. Used for window partition, shifted windows, and attention mask design.
- Dao et al., FlashAttention. Used for IO-aware exact attention discussion and the distinction between exact attention IO optimization and sparse/approximate attention.
