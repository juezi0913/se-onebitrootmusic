# SE-RootMUSIC for One-Bit DOA Estimation

This repository contains the current `SE-RootMUSIC` implementation for one-bit direction-of-arrival estimation.

## Method summary

The method keeps one-bit root-MUSIC as the estimation backbone and enhances it through:

1. structured one-bit covariance reconstruction,
2. closed-loop covariance refinement,
3. subspace enhancement before root-MUSIC readout,
4. confidence-gated global one-bit consistency refinement.

Compared with the earlier pair-wise correction version, this version removes the explicit single-pair repair stage and replaces it with a low-dimensional global refinement over all sources.

## Repository structure

- `src/se_rootmusic_1bit_doa_estimator.m`: main proposed method
- `src/root_music_doa.m`: root-MUSIC readout routine
- `experiments/demo_compare_paper8_with_v3.m`: experiment driver for the eight benchmark settings
- `experiments/merge_se_rootmusic_with_baselines.m`: merge script that combines the new method with previously saved baseline results
- `figures/`: refreshed final figures
- `results/results_20260424_170912.mat`: merged result file used to generate the current final plots

## Current result set

The `figures/` folder contains the refreshed eight benchmark plots built from:

- previous baseline runs
- the current full-run `SE-RootMUSIC` results

These figures are intended for paper drafting and result comparison.
