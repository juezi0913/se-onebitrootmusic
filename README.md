# SE-RootMUSIC for One-Bit DOA Estimation

This repository contains the current paper-version `SE-RootMUSIC` implementation for one-bit direction-of-arrival estimation.

## Method summary

The method keeps one-bit root-MUSIC as the estimation backbone and enhances it through:

1. quantization-consistent and model-consistent covariance correction,
2. Toeplitz / Hermitian / PSD structural projection,
3. explicit structured subspace projection of the form
   `Us*Lambda_s*Us' + sigma2*Un*Un'`,
4. confidence-gated anchored global one-bit consistency refinement.

Compared with the earlier pair-wise correction version, this version removes the explicit single-pair repair stage and replaces it with a low-dimensional global refinement over all sources. The current paper configuration also uses fixed experiment parameters instead of scene-adaptive tuning in the main SE pipeline.

## Repository structure

- `src/se_rootmusic_1bit_doa_estimator.m`: previous SE-RootMUSIC implementation
- `src/se_rootmusic_cleanproj_1bit_doa_estimator.m`: current clean-projection paper version
- `src/root_music_doa.m`: root-MUSIC readout routine
- `src/hp_crb_stochastic_ula.m`: CRB reference utility
- `experiments/demo_compare_paper8_with_v3.m`: experiment driver for the eight benchmark settings
- `experiments/merge_se_rootmusic_with_baselines.m`: merge script that combines the new method with previously saved baseline results
- `figures/`: refreshed final figures
- `results/results_20260427_162032.mat`: current merged result file used to generate the latest final plots

## Current result set

The `figures/` folder contains the refreshed eight benchmark plots built from:

- previous baseline runs
- the current full-run clean-fixed `SE-RootMUSIC` results

These figures are intended for paper drafting and result comparison. The corresponding current merged result file is:

- `results/results_20260427_162032.mat`
