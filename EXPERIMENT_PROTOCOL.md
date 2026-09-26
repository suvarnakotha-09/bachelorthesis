# Reproducible experiment protocol

This document is the source of truth for the comparison implemented in
`traffic_experiment.py` and presented in the experiment notebook.

## Comparison contract

- **Target:** `traffic_speed` from `METR_LA_with_Weather_5min.csv`.
- **Primary feature set:** traffic speed only. Weather columns are retained in
  the source data but excluded from the primary comparison so that TimesFM,
  DSS-softmax, and LSTM receive the same target and history information.
- **Context:** 24 observations (two hours at five-minute resolution).
- **Forecast horizon:** one observation (five minutes). Every model is scored at
  the same target timestamps.
- **Split:** chronological 70% train, 10% validation, 20% test. No random
  shuffling is used to construct or split windows.
- **Scaling:** `MinMaxScaler` is fitted on the raw training target only. The
  fitted training parameters are then used to transform validation and test
  data. TimesFM uses its own documented input normalization and receives raw
  traffic values.
- **Optimization:** Adam, learning rate `1e-3`, gradient clipping at `1.0`,
  validation-MAE early stopping, patience `5`, a maximum of `20` epochs, and
  restoration of the best validation state.
- **Batch size:** 64 for supervised models; DataLoader order is preserved
  (`shuffle=False`). TimesFM uses a recorded inference batch size of 32.
- **Randomness:** Python, NumPy, and PyTorch use seed `42`; the seed and
  package versions are written with every run.

## Window construction

For a target index `t`, a supervised model receives the scaled history
`x[t-24:t]` and predicts `y[t:t+1]`. Training targets end before the validation
boundary, validation targets end before the test boundary, and test contexts may
use only observations that precede their target. This avoids both target
leakage and scaler leakage.

The run metadata records the exact number of raw rows and train/validation/test
windows. The full-data split for this repository is 30,240 rows, 21,144
training windows, 3,024 validation windows, and 6,048 test windows.

## DSS-softmax baseline

The supervised diagonal state-space baseline follows the canonical DSS-softmax
recurrence of Gupta, Gu, and Berant (2022), wrapped in a traffic-specific
PyTorch module. For a fixed context length `L`, hidden channel `h`, complex
mode `i`, positive learned timescale `Δ_h`, and input `u_h[t]`:

\[
    a_{hi}=e^{\lambda_i\Delta_h},
    \qquad
    b_{hi}=\frac{e^{\lambda_i\Delta_h}-1}
    {\lambda_i\left(e^{L\lambda_i\Delta_h}-1\right)},
\]

\[
    x_i[t]=a_{hi}x_i[t-1]+b_{hi}u_h[t],
    \qquad
    y_h[t]=\operatorname{Re}\sum_i W_{hi}x_i[t].
\]

The length-normalized denominator is the DSS-softmax discretization; ordinary
ZOH coefficients without it are a related but different diagonal SSM. The
wrapper adds feedthrough, GELU, residual paths, layer normalization, and a
one-step projection. It uses 64 hidden channels, 64 complex modes, and two
blocks. The complex eigenvalues are initialized from HiPPO-D, output weights
are standard-normal real/imaginary values, and `log(Δ_h)` is initialized in the
canonical log-uniform range. The trainable count is measured from the actual
PyTorch module rather than copied from the original paper.

The original DSS paper evaluates long-range sequence benchmarks rather than
METR-LA. This traffic wrapper is therefore described as a DSS-softmax traffic
adaptation, not as a claim that every detail of the original benchmark code is
reproduced. The earlier custom gated recurrence remains available in the code
under explicit `DSS-inspired` aliases for a separate ablation.

## Models and timing

- **LSTM:** two layers, hidden size 64, dropout 0.2.
- **DSS-softmax:** two diagonal state-space blocks, model width 64, 64 complex
  modes, HiPPO-D initialization, residual connections, layer normalization, and
  a one-step projection.
- **TimesFM:** the pretrained TimesFM 2.5 PyTorch checkpoint, used zero-shot
  with `torch_compile=False`; it is not presented as a model trained under the
  same optimization regime as the supervised models.

Training time, one-step inference latency per 1,000 test windows, device, and
peak process memory are recorded in `outputs/efficiency.csv`. Process RSS is
sampled during each sequential model stage and is not an isolated per-model
memory footprint. Accuracy and computational claims must use these records
rather than unsupported adjectives.

## Outputs

A full run writes the following reproducibility records:

- `outputs/metrics.csv` — accuracy, sample counts, and task metadata;
- `outputs/efficiency.csv` — timing, parameter, device, and memory measurements;
- `outputs/run_metadata.json` — versions, hardware, seed, split, scaler, and
  model metadata;
- `outputs/figures/metrics_by_unit.png` — accuracy overview;
- `outputs/figures/mae_rmse_speed.png` — MAE/RMSE in original speed units;
- `outputs/figures/mape_percent.png` — MAPE on a separate percentage axis;
- `outputs/figures/forecast_comparison.png` — aligned one-step predictions.

The first TimesFM run downloads its checkpoint. A CPU run is valid but can be
substantially slower than a CUDA run; hardware and device are therefore part of
the recorded experimental contract.
