# Mono-QPD Project Guide

This document provides context for AI agents working on the Mono-QPD project.

## Project Overview

**Mono-QPD**: Monocular Depth Estimation with Quad-Pixel Data

- **Language**: Python
- **Framework**: PyTorch
- **Main Purpose**: Deep learning research on depth estimation using quad-pixel sensor data

## Development Environments

### 1. Local Development (Windows WSL)
- **Location**: `/mnt/d/Mono+Dual/mono-qpd/mono-qpd-simple`
- **Purpose**: Code testing, debugging, small experiments
- **Usage**: Direct script execution
  ```bash
  # Training
  python train_mono_qpd.py --exp_name LocalExp0206QPDNet
  
  # Evaluation
  python evaluate_mono_qpd.py --exp_name LocalExp0206QPDNet --ckpt_epoch latest --eval_datasets DPD_Disp
  ```
- **VSCode Debug**: Use `.vscode/launch.json` configurations
- **Git Branch**: Usually work on feature branches

### 2. TSUBAME Supercomputer (Remote)
- **Purpose**: Large-scale training, batch experiments
- **Job Management**: Uses `runsync` system
- **Usage**: 
  ```bash
  # Submit job via runsync
  python runsync/runsync_script.py --type train --run_setting_name Exp0206QPDNet --run_script True
  ```
- **Important**: `runsync` is ONLY for TSUBAME, NOT for local testing
- **Config Location**: `runsync/presets/run_settings.py`

## Key Files and Directories

### Core Model Files
- `mono_qpd/mono_qpd.py` - Main MonoQPD model class
  - **Important**: `self.da_v2` can be `None` when `include_da_v2: bool = False`
- `mono_qpd/QPDNet/qpd_net.py` - QPDNet architecture
- `mono_qpd/QPDNet/Quad_datasets.py` - Dataset handling

### Training & Evaluation
- `train_mono_qpd.py` - Training script (direct execution)
- `evaluate_mono_qpd.py` - Evaluation script (direct execution)
- `exp_args_settings/train_settings/train_settings.py` - Experiment configurations

### TSUBAME-specific (Remote Only)
- `runsync/` - Job submission system for TSUBAME supercomputer
- `runsync/presets/run_settings.py` - Experiment presets for remote execution
- `runsync/runsync_script.py` - Job submission script

### Configuration
- `.vscode/launch.json` - VSCode debug configurations for local testing
- `.opencode/oh-my-opencode.json` - OpenCode agent configurations

## Experiment Configurations

Experiments are defined as dataclasses in `runsync/presets/run_settings.py`:

### Example: Exp0206QPDNet
```python
@dataclass
class Exp0206QPDNet(BaseConfig):
    num_steps: int = 200_000
    batch_size: int = 6
    include_da_v2: bool = False  # ⚠️ Important: da_v2 is disabled
    feature_converter: str = 'interp'
    val_datasets: Tuple[str] = ('DPD-Disp',)
```

### Local vs Remote Naming Convention
- `LocalExp*`: Local testing experiments (small batch size, debug mode)
- `Exp*`: Production experiments for TSUBAME (large batch size)

## Common Issues and Solutions

### 1. Conda Environment Setup
- **Issue**: `CondaError: Run 'conda init' before 'conda activate'` and `libmambapy.QueryFormat` errors
- **Solution**: These are conda/libmamba initialization warnings. **IGNORE THEM.** They don't affect code execution.
- **Workaround**: When using conda in bash, always use:
  ```bash
  eval "$(conda shell.bash hook)" && conda activate mono-qpd
  ```
- **Note**: The mono-qpd environment is already configured with all required packages

### 2. AttributeError: 'NoneType' object has no attribute 'load_state_dict'
- **Cause**: `model.da_v2` is `None` when `include_da_v2: bool = False`
- **Solution**: Always check `if model.da_v2 is not None:` before calling `model.da_v2.load_state_dict()`
- **Fixed in**: `evaluate_mono_qpd.py:1296` (2026-02-13)

### 3. Auto-formatting Changes
- **Issue**: Black/Ruff formatter applies unwanted changes
- **Solution**: Disabled in `.opencode/oh-my-opencode.json`
  ```json
  "lsp": {
    "auto_format": false,
    "format_on_save": false
  }
  ```

## Workflow Patterns

### Local Testing Workflow
1. Edit code in local repository
2. Test with VSCode debug configuration (F5)
3. Use `LocalExp*` configurations (small batch size)
4. Verify results
5. Commit changes

### TSUBAME Deployment Workflow

#### Training on TSUBAME
1. Ensure local tests pass
2. Push to remote branch
3. Submit training job via runsync:
   ```bash
   python runsync/runsync_script.py --type train --run_setting_name Exp0206QPDNet --run_script True
   ```
4. Monitor job status on TSUBAME
5. Pull results back for analysis

#### Evaluation on TSUBAME (Standalone)
When you want to evaluate a trained model without training:
```bash
python runsync/runsync_script.py \
  --type eval \
  --run_setting_name Exp0114LongQPDv2 \
  --run_script True \
  --eval_run_time 01:30:00 \
  --eval_datasets QPD-Test QPDv2-Test DP5K-Test DPD_Disp
```

**For ReplaceAiF models** (uses QPD_AiF dataset):
```bash
python runsync/runsync_script.py \
  --type eval \
  --run_setting_name Exp0206ReplaceCenterAiF \
  --run_script True \
  --eval_run_time 00:30:00 \
  --eval_datasets QPD_AiF
```

**Parameters:**
- `--type eval`: Evaluation mode (not training)
- `--run_setting_name`: Experiment name (must have trained checkpoint)
- `--run_script True`: Execute immediately
- `--eval_run_time`: Job time limit (format: HH:MM:SS)
- `--eval_datasets`: Space-separated list of datasets to evaluate

**Available Datasets:**
- `QPD-Test`: QPD test set (~5 min on TSUBAME)
- `QPDv2-Test`: QPD version 2 test set
- `DP5K-Test`: DP5K test set
- `QPD_AiF`: ReplaceAiF synthetic dataset (~5 min on TSUBAME)
- `DPD_Disp`: DPD disparity dataset (~3 min on TSUBAME)
- `DP119`: DP119 dataset
- `Real-QPD`: Real QPD data
- `QPD-FStop-*`: Various f-stop configurations

**Evaluation Time Reference (TSUBAME):**
- `QPD_AiF`: ~5 minutes (recommend 10 min with margin)
- `QPD-Test + DPD_Disp`: ~8 minutes (recommend 15 min with margin)
- Single large dataset: 15-30 minutes depending on size
- Multiple datasets: Add times + 5-10 min margin

## Important Notes

### DO NOT:
- ❌ Use `runsync` for local testing
- ❌ Run production experiments (`Exp*`) on local machine
- ❌ Assume `model.da_v2` is always available
- ❌ Commit auto-formatted files without review

### DO:
- ✅ Use `LocalExp*` configs for local testing
- ✅ Check `include_da_v2` flag before accessing `model.da_v2`
- ✅ Test locally before deploying to TSUBAME
- ✅ Use `.vscode/launch.json` for debugging
- ✅ Keep runsync configs separate from local testing

## Git Branch Strategy

- `main` / `master`: Stable production code
- `exp*/*`: Experiment branches for specific features
- Current active branch: `exp0109/new-synthetic`

## Experiment Results

See [EXPERIMENTS.md](./EXPERIMENTS.md) for benchmark results and comparisons.

## Contact & Resources

- TSUBAME Documentation: [TSUBAME Portal]
- Internal Wiki: [Project Wiki Link]
- Issue Tracker: [GitHub Issues / Internal Tracker]

---

**Last Updated**: 2026-02-13
**Maintained by**: OpenCode AI Agent
