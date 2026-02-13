# Experiment Results

This document tracks experimental results for AI agent reference.

---

## Benchmark Results (2026-02-13)

### QPD Dataset

| Method | w/o noise |  |  |  | w/ noise |  |  |  |
|--------|-----------|------|-------|-------|----------|------|-------|-------|
|        | EPE↓ | RMSE↓ | AI(1)↓ | AI(2)↓ | EPE↓ | RMSE↓ | AI(1)↓ | AI(2)↓ |
| RAFT-Stereo (QPD) | 0.0298 | 0.0818 | 0.0295 | 0.0807 | 0.1297 | 0.2634 | 0.1252 | 0.2343 |
| Baseline (QPD) | 0.0269 | 0.0768 | 0.0266 | 0.0759 | 0.0882 | 0.1820 | 0.0851 | 0.1683 |
| Depth Anything V2 | - | - | 0.1926 | 0.3319 | - | - | 0.2168 | 0.3706 |
| **FMDP (QPD)** | 0.0259 | 0.0725 | 0.0256 | 0.0712 | 0.0526 | 0.1176 | 0.0500 | 0.1100 |
| FMDP (QPDv2) | 0.0445 | 0.0992 | 0.0416 | 0.0934 | 0.0740 | 0.1380 | 0.0679 | 0.1262 |
| FMDP (Mixed) | 0.0334 | 0.0880 | 0.0327 | 0.0867 | 0.0575 | 0.1187 | 0.0547 | 0.1143 |

### QPDv2 Dataset

| Method | EPE↓ | RMSE↓ | AI(1)↓ | AI(2)↓ |
|--------|------|-------|--------|--------|
| FMDP (QPD) | 1.5110 | 2.0720 | 0.6746 | 1.0220 |
| **FMDP (QPDv2)** | 0.1543 | 0.3831 | 0.1468 | 0.3669 |
| FMDP (Mixed) | 0.1881 | 0.4458 | 0.1735 | 0.4156 |

### Real-World Datasets (DP-Disp & DP5K)

| Method | DP-Disp |  |  | DP5K |  |  |
|--------|---------|------|---------|------|------|---------|
|        | AI(1)↓ | AI(2)↓ | 1-\|ρs\|↓ | AI(1)↓ | AI(2)↓ | 1-\|ρs\|↓ |
| RAFT-Stereo (QPD) | 0.0339 | 0.0575 | 0.2309 | 0.0136 | 0.0990 | 0.1011 |
| Baseline (QPD) | 0.0302 | 0.0547 | 0.2275 | 0.0232 | 0.1063 | 0.2102 |
| Depth Anything V2 | 0.0332 | 0.0678 | 0.2353 | 0.0134 | 0.0949 | 0.0574 |
| FMDP (QPD) | 0.0256 | 0.0466 | 0.2232 | 0.0164 | 0.1133 | 0.1044 |
| FMDP (QPDv2) | 0.0284 | 0.0509 | 0.2231 | 0.0125 | 0.0969 | 0.0765 |
| **FMDP (Mixed)** | 0.0262 | 0.0484 | 0.2210 | 0.0120 | 0.0964 | 0.0734 |

---

## Exp0206QPDNet Evaluation (2026-02-13)
**Model**: QPDNet without Depth Anything V2 (`include_da_v2: False`)  

### QPD-Test Results by Epoch

| Epoch | EPE↓ | RMSE↓ | AI(1)↓ | AI(2)↓ | SI↓ |
|-------|------|-------|--------|--------|-----|
| 250 | 0.0284 | 0.0816 | 0.0282 | 0.0807 | 0.0242 |
| 255 | 0.0283 | 0.0812 | 0.0280 | 0.0803 | 0.0234 |
| 260 | 0.0282 | 0.0811 | 0.0280 | 0.0802 | 0.0234 |
| **265** | **0.0282** | **0.0810** | **0.0279** | **0.0801** | **0.0232** |

### DP-Disp Results by Epoch

| Epoch | AI(1)↓ | AI(2)↓ | 1-\|ρs\|↓ |
|-------|--------|--------|-----------|
| 250 | 0.0291 | 0.0527 | 0.2288 |
| 255 | 0.0294 | 0.0534 | 0.2291 |
| 260 | 0.0293 | 0.0530 | 0.2287 |
| **265** | **0.0293** | **0.0530** | **0.2284** |

---

## Key Findings

1. **FMDP (QPD)**: Best performance on QPD dataset (both w/ and w/o noise)
2. **FMDP (QPDv2)**: Best performance on QPDv2 dataset
3. **FMDP (Mixed)**: Most balanced performance across all datasets
4. **Noise robustness**: FMDP models significantly outperform baselines under noisy conditions

---

## Metric Definitions

- **EPE**: End-Point Error (lower is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **AI(n)**: Absolute error at threshold n (lower is better)
- **1-|ρs|**: 1 minus absolute Spearman correlation (lower is better = higher correlation)

---

*Last updated: 2026-02-13 14:00 JST*
