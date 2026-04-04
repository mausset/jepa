# JEPA World Models

Research codebase for latent world models using Joint Embedding Predictive Architecture.

## Project structure

```
configs/                  Hydra config hierarchy
  config.yaml             Root config (defaults: local cluster)
  cluster/                Hardware configs (local, hopper, thin)
  experiment/             Full experiment configs (toy_*, vits)
  sweep/                  Hyperparameter sweep definitions
  envs/                   Environment specifications
src/jepa/
  train.py                Training loop, loss computation, checkpointing
  launch.py               Sweep launcher (local or SLURM via submitit)
  models/
    jepa.py               Main model: encoder -> predictor pipeline
    encoder.py            ViT (t/s/b/l) and ConvNeXt (t/s/b/l) encoders
    predictor.py          Temporal predictor with encoder half + conditioned predictor half
    action_decoder.py     Transformer action decoder (discrete or continuous)
    modules.py            SwiGLUFFN, attention masks
  datasets/
    builder.py            Dataset factory (toy_env, video, craftax_online)
    toy_env_dataset.py    HDF5 offline trajectories with augmentation
    video_dataset.py      NVIDIA DALI video pipeline
    craftax_online.py     Online JAX environment stepping
  envs/
    toy_envs.py           Environment wrappers (Gym, MiniGrid, Sokoban, PushT, Craftax)
    collect_toy_env_data.py  HDF5 dataset generation from environments
  losses/
    sigreg.py             Epps-Pulley normality test + sliced projection (prevents collapse)
  utils/
    distributed.py        DDP setup, all_gather, all_reduce_cov
    helpers.py            rankme, spectrum, MeanMetric, attention masks
    scheduler.py          TrapezoidSchedule, WarmupCosineSchedule
  planning/
    base_planner.py       Abstract planner interface
    cem.py                Cross-entropy method planner (stub)
```

## Architecture

The model (`JEPA`) chains an encoder and a temporal predictor:

1. **Encoder**: processes each frame independently, outputs register embeddings. ViT uses axial RoPE + register tokens; ConvNeXt uses pooled backbone features.
2. **Predictor**: split into an encoder half (PlainBlock, no conditioning) and a predictor half (conditioned blocks). The predictor half supports two conditioning modes:
   - `adaln` (default): AdaLN modulation from quantized latent
   - `add`: latent added to state before plain blocks
3. **FSQ bottleneck** (optional): discretizes encoder output into codes via Finite Scalar Quantization.
4. **Action decoder** (optional): transformer predicting actions from states. Gradients are detached from the encoder (action decoder doesn't update encoder).

### Predictor modes

- `mean`: single predictor pass with null (zero) latent. Predicts average future.
- `latent`: single pass with FSQ latent. Predicts conditioned future.
- `residual`: batched dual pass through shared blocks. Mean pass (null latent) + residual pass (FSQ latent). Loss is averaged over the two MSE terms.

## Loss

`total = lambda * sigreg + mse_terms + action_loss`

- **SigReg**: Epps-Pulley normality test on random projections of encoder states. Prevents representation collapse. Uses differentiable `all_reduce` for DDP.
- **MSE**: prediction error against encoder targets. In residual mode, the two MSE losses (mean + conditional) are averaged.
- **Action loss**: MSE (continuous) or cross-entropy (discrete).

## Running experiments

```bash
# Single run
python -m jepa.launch --sweep-name <name> +experiment=<exp>

# With overrides
python -m jepa.launch --sweep-name test +experiment=toy_pusht training.total_steps=5000

# Multiple seeds
python -m jepa.launch --sweep-name test +experiment=toy_pusht --seeds 3
```

The launcher creates per-run configs in `experiments/<sweep>/configs/`, runs via `torchrun`, and saves results (config + final val metrics + train time + git hash) to `experiments/<sweep>/results/<run_id>.json`.