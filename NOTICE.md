# Notices

The MIT licence in `LICENSE` covers the code in this repository. It does not
cover the data the code operates on, which carries its own terms.

## AMLworld

Every dataset artefact here is derived from AMLworld, IBM's synthetic
anti-money-laundering transaction dataset (Altman et al., NeurIPS 2023
Datasets and Benchmarks), published at
<https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml>
under the **Community Data License Agreement, Sharing, Version 1.0**
(confirmed from the listing, 2026-08-10).

CDLA-Sharing is copyleft: anyone who publishes data derived from AMLworld has
to publish it under the same terms. That reaches the coreset indices, the
Horvitz-Thompson weights, the labels and typologies, the GFP tensors and the
serialised prompts, all of which are derived data. It does not reach this
code, which is MIT, and by the agreement's carve-out for computational results
it does not reach the aggregate metrics in `results/` or the supervised model
weights trained on the data.

This repository does **not** redistribute the AMLworld transaction CSVs. It
ships coreset indices, Horvitz-Thompson weights, labels and typologies, all
derived. `scripts/00_download_amlworld.py` fetches the originals from Kaggle
under the user's own account.

## Elliptic

The Elliptic Data Set backs the real-data check in Appendix D. It is published
at <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set> under
**CC BY-NC-ND 4.0** (confirmed from the listing, 2026-08-10). NoDerivatives
forbids distributing adapted material at all, under any notice, so this
repository is **code only** for Elliptic: `scripts/elliptic/` downloads and processes the data locally, and
only aggregate result statistics are committed. No Elliptic tensors, coresets
or serialisations are shipped.

## Prior work reused

- The typed-graph serialisation and the ICL-FS and ICL-ZS prompt text follow
  the protocol of Pirmorad et al. (2025). `prompts/icl_fs.j2` and
  `prompts/icl_zs.j2` are that protocol; `prompts/icl_v.j2` adds this paper's
  verification step and is the only prompt contribution here.
- The Graph-Feature-Preprocessor is IBM Snap ML.
- The GCPAL baseline is a reimplementation of Lu et al. (2024); no official
  code was available.
- Baseline B5 implements Fogliato et al. (ECCV 2024).
- B6 follows Leskovec and Faloutsos (2006); B7 follows Gao et al.

## Evaluated model weights

The seven open-weight models evaluated here are not redistributed. Each is
governed by its own model card on Hugging Face.

## Supervised baseline weights

The LightGBM, XGBoost and GCPAL checkpoints we trained are ours and are
released under the MIT licence, alongside the dataset on Hugging Face.
