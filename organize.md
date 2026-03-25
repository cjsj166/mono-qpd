# Codebase Cleanup & README Plan

## Completed

- [x] README.md rewritten (English, public release format)
  - FMDP: MVA 2025 official implementation
  - QPD-only sections for public release
  - Appendix "Internal Handover Only" section for QPD+QPDv2 and TSUBAME

## Files to Delete

| Target | Reason |
|--------|--------|
| `.opencode/` | AI agent internal docs, not for public |
| `exp_utils/` | Personal utilities (PPT layout, result aggregation, plotting) |
| `evaluate_multiple_models.py` | Personal multi-model evaluation script |
| `train_script_composer.py` | runsync helper script |
| `eval_script_composer.py` | runsync helper script |
| `qpd-env.yaml` | Duplicate env file (env.yml is the canonical one) |
| `conda-freeze-results.txt` | Personal pip freeze output |

## Files to Keep

| Target | Reason |
|--------|--------|
| `runsync/` | Referenced in README Appendix (TSUBAME handover) |
| `run_settings.py` configs (all) | Keep all experiment configs as-is |
| `.vscode/` | Keep for now |
| `scripts/` | Keep |
| `env.yml` | Canonical environment file |

## README Structure Decisions

- **Public sections**: Environment Setup, Pretrained Checkpoints (DA-V2 via GitHub link, FMDP), Dataset (QPDNet GitHub link), Training (QPD), Evaluation (QPD)
- **Handover-only section** (remove before public release): QPD+QPDv2 dataset/training/eval/checkpoint, TSUBAME usage via runsync
- No Results/Citation/Acknowledgement sections
- Dataset links: QPDNet GitHub (https://github.com/Zhuofeng-Wu/QPDNet), not direct Dropbox
- DA-V2 checkpoint: Depth Anything V2 GitHub (https://github.com/DepthAnything/Depth-Anything-V2)
- Environment setup: single `conda env create -f env.yml` command

## Key Experiment Configs (for reference)

- `Exp0825FMDP` — QPD training (public, default)
- `Exp0121MixedDataset` — QPD+QPDv2 mixed training (handover only)
- `Exp0206QPDNet` — QPDNet without Depth Anything V2
