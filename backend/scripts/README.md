## Training scripts overview

This directory contains the training and export pipeline for two residual models that sit on top of a simple physics baseline for concrete compressive strength:

- A **Gaussian Process residual model** (`physics_gp.py` + `train.py` + `prep_gp_ckpt.py`)
- A **Neural network residual model** (`physics_nn.py` + `train.py` + `prep_nn_ckpt.py`)

Both models learn a **residual correction** to a physics-based baseline prediction (a Hirsch-style mixture rule serves as an initial placeholder), using the same feature set and scalers. The exported `ONNX` models expose only the **residual predictor**; the physics baseline is recomputed separately in production and added back in.

The common physics baseline and feature definitions live in `physics_shared.py`.

---

## Shared physics baseline (`physics_shared.py`)

### Feature set

The models use a fixed feature ordering defined by `FEATURE_COLS`:

- **mortar_splitting_strength_mpa**
- **mortar_shrinkage_in**
- **mortar_flexural_strength_mpa**
- **mortar_slump_in**
- **mortar_compressive_strength_mpa**
- **mortar_poissons_ratio**
- **rock_compressive_strength_mpa**
- **rock_size_in**
- **rock_density_lb_ft3**
- **rock_specific_gravity**
- **rock_ratio**

Every ML model (NN or GP) expects its inputs to be these 11 features, in this exact order, typically **standard‑scaled** by `scaler_x`.

### Physics baseline model

- **Function**: `base_physics_model(row)`  
  - Currently implemented as `hirsch_row`, a Hirsch-style mixture rule:
    - Takes:
      - `mortar_compressive_strength_mpa` (fm)
      - `rock_compressive_strength_mpa` (fr)
      - `rock_ratio` (r)
    - Returns a scalar predicted compressive strength in MPa:
      - A weighted combination of mortar and rock strengths, plus an interaction term parameterized by `eta` (default `0.1`).

This physics prediction is used as a **baseline**. The ML models learn a **residual**:

$$
\text{final\_prediction} = \text{base\_physics\_prediction} + \text{predicted\_residual}
$$

### PhysicsModel interface and combination helper

- **Abstract interface**: `PhysicsModel`
  - Requires:
    - `predict_residual(x: torch.Tensor) -> (mean_res, std_res)`
    - `scaler_x` and `scaler_y` attributes providing `transform` / `inverse_transform`.

- **Helper**: `combine_physics_with_residual(df_infer, model)`
  - Copies `df_infer`,
  - Computes `physics_base` via `base_physics_model` row‑wise,
  - Builds `X` by selecting `FEATURE_COLS`,
  - Scales `X` with `model.scaler_x`,
  - Calls `model.predict_residual` to get mean and std of the residual,
  - Produces:
    - `pred_mpa = physics_base + mean_res`
    - `pred_lo_mpa = pred_mpa - std_res`
    - `pred_hi_mpa = pred_mpa + std_res`
  - Returns a DataFrame with `physics_base`, `pred_mpa`, `pred_lo_mpa`, `pred_hi_mpa`, `residual`.

This is how the training scripts validate models after training.

---

## Neural Net residual model (`physics_nn.py`)

### Architecture and training

- **Class**: `PhysicsNN(L.LightningModule)`
- **Inputs**:
  - `input_dim`: number of features (should be `len(FEATURE_COLS) == 11`)
  - `scaler_x`, `scaler_y`: `sklearn`-style scalers for inputs and residual outputs
  - `hidden_dim` (default `64`), `n_hidden_layers` (default `4`)

- **Network**:
  - A simple feed‑forward MLP:
    - Input layer: `Linear(input_dim, hidden_dim)` + `ReLU`
    - `n_hidden_layers` blocks: `Linear(hidden_dim, hidden_dim)` + `ReLU`
    - Output layer: `Linear(hidden_dim, 1)`
  - Stored as `self.model = torch.nn.Sequential(...)`.

- **Forward**:
  - `forward(x)` just calls `self.model(x)` and returns a `(batch, 1)` tensor of **scaled residuals**.

- **Training / validation**:
  - `training_step`:
    - Receives `(x, y)` where `y` is the **scaled residual target**.
    - Computes `y_hat = self(x).squeeze(-1)` and MSE loss vs `y`.
    - Logs `"train_loss"` to Lightning.
  - `validation_step`:
    - Same loss, plus `"val_rmse"` = `sqrt(MSE)` for checkpointing.

- **Optimizer**:
  - `configure_optimizers` uses `Adam` with `lr=1e-3`.

- **Residual prediction API**:
  - `predict_residual(x_scaled)`:
    - Assumes `x_scaled` is already scaled with `scaler_x`.
    - Runs `self(x_scaled)` on `self.device` with `no_grad`.
    - Converts to numpy and **inverse‑transforms** through `scaler_y` to get residuals in MPa.
    - Returns:
      - `mean_res` (1D numpy array of residuals in MPa),
      - `std_res` (zeros array, since NN does not model uncertainty).

### Training script and checkpoint (`train.py` + `prep_nn_ckpt.py`)

- **Training** (`train.py` with `--model nn`):
  - Loads the concrete dataset from `../../frontend/data` via `lightning_dataloader.py`.
  - Builds train/val splits, and `scaler_x`, `scaler_y`.
  - Instantiates `PhysicsNN(input_dim=len(FEATURE_COLS), scaler_x=sx, scaler_y=sy)`.
  - Uses Lightning’s `ModelCheckpoint` callback to save the best model in `checkpoints/` based on `val_rmse`.
  - Writes the best checkpoint path to `best_ckpt_path.txt`.
  - After training, loads the best checkpoint and runs a small validation inference via `combine_physics_with_residual`.

- **Checkpoint prep and ONNX export** (`prep_nn_ckpt.py`):
  - Reads `best_ckpt_path.txt`.
  - Loads the Lightning checkpoint on CPU.
  - Extracts hyperparameters:
    - `input_dim`, `hidden_dim`, `n_hidden_layers`
  - Rebuilds a **vanilla `nn.Sequential`** with the same layout as `PhysicsNN.model`.
  - Strips `"model."` prefixes from the `state_dict` and loads them into the vanilla model.
  - Loads `scaler_x` and `scaler_y` from `scalers/scaler_x.joblib` and `scalers/scaler_y.joblib` if available.
  - Loads realistic concrete/mortar/rock examples from `../../frontend/data/*.json`,
    builds feature vectors in `FEATURE_COLS` order, optionally scales with `scaler_x`,
    and runs:
    - **PyTorch inference** to get scaled residuals,
    - Inverse transforms with `scaler_y` to get residuals in MPa,
    - Adds the physics baseline via `base_physics_model` and prints final strengths vs ground truth.
  - **ONNX export**:
    - Exports `vanilla_model` to `physics_nn.onnx` with:
      - `opset_version=14`
      - Dynamic batch dimension on input and output.
  - **ONNX verification and benchmarking**:
    - Loads `physics_nn.onnx` with `onnxruntime`.
    - Runs inference on the same scaled inputs.
    - If `scaler_y` is present, inverse‑transforms outputs and compares PyTorch vs ONNX residuals, printing:
      - Per‑example PyTorch vs ONNX residual and absolute difference,
      - Max difference, with thresholds to classify as “match within numerical precision” vs “small differences”.
    - Benchmarks 1000 iterations of:
      - PyTorch `vanilla_model` on a batch,
      - ONNX Runtime session on the same batch,
      - Prints per‑batch latency and speedup.
  - **Standalone production example**:
    - Shows how to:
      - Prepare realistic unscaled inputs,
      - Scale with `scaler_x`,
      - Run ONNX, inverse‑transform with `scaler_y`,
      - Add physics baseline to get final compressive strength.

**Key caveats for NN ONNX export**:

- The ONNX model only predicts the **scaled residual**; production must:
  - Apply `scaler_x.transform` to features,
  - Run ONNX,
  - Apply `scaler_y.inverse_transform` to get residual MPa,
  - Recompute the physics baseline and add the residual.
- No predictive uncertainty: `std_res` is always zero.

---

## Gaussian Process residual model (`physics_gp.py`)

### Architecture and training

The GP model is an **Exact GP** over the residuals, with a Matern‑5/2 ARD kernel, trained using GPyTorch.

- **Core GP**: `ExactResidualGP(gpytorch.models.ExactGP)`
  - Mean function: `ZeroMean`
  - Covariance: `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_x.shape[1]))`
  - `forward(x)` returns a `MultivariateNormal(mean_x, covar_x)` representing the residual distribution in **scaled output space**.

- **Lightning module**: `PhysicsGPR(L.LightningModule)`
  - Holds:
    - `scaler_x`, `scaler_y`
    - A `GaussianLikelihood`
    - The `ExactResidualGP` instance (`self.gp`)
    - An `ExactMarginalLogLikelihood` (`self.mll`)
    - Cached training inputs `train_x`, `train_y` in scaled space (used for exact inference and restarting from checkpoints).

#### Special checkpoint handling

Exact GPs have internal state beyond standard module parameters (e.g. prediction strategy). `PhysicsGPR` implements custom save/load logic:

- **on_save_checkpoint**:
  - Saves:
    - GP parameters prefixed with `"gp."`,
    - MLL parameters prefixed with `"mll."`,
    - Other Lightning module parameters (excluding `"gp."` / `"mll."`),
    - `scaler_x`, `scaler_y`, `train_x`, `train_y`.

- **on_load_checkpoint**:
  - Restores `train_x`, `train_y`, `scaler_x`, `scaler_y` from the checkpoint.

- **load_state_dict override**:
  - Splits incoming `state_dict` into:
    - `gp_state` (keys starting with `"gp."`),
    - `mll_state` (keys starting with `"mll."`),
    - `other_state` (everything else).
  - Loads `other_state` via `super().load_state_dict(strict=False)`.
  - If `train_x`/`train_y` are present, it calls `setup_gp(train_x, train_y)` to rebuild `self.gp` and `self.mll`, then loads `gp_state` and `mll_state` into those objects.

- **training_step**:
  - On the first batch, calls `setup_gp(x, y)` to build the `ExactResidualGP` and `ExactMarginalLogLikelihood`.
  - Computes the negative log marginal likelihood loss: `loss = -mll(gp(x), y)`.
  - Logs `"train_nll"`.

- **validation_step**:
  - Ensures the GP is initialized (`setup_gp` if needed).
  - Computes predictive mean on the validation batch and returns:
    - `val_rmse` (RMSE of mean vs targets),
    - `val_nll` (negative log predictive probability per point).
  - These metrics are logged; `train.py` uses `val_rmse` for checkpointing.

- **predict_residual(x_scaled)**:
  - Runs `self.gp` + `self.likelihood` in `fast_pred_var()` mode.
  - Computes predictive mean and stddev in **scaled space**.
  - Inverse‑transforms both through `scaler_y` to get residual mean and std in MPa (for use with `combine_physics_with_residual`).

### Training script (`train.py` with `--model gp`)

- `train.py` chooses `PhysicsGPR(sx, sy)` when `--model gp`.
- The data loading and scaler creation are the same as for the NN.
- Training runs for up to 200 epochs, with `ModelCheckpoint` monitoring `"val_rmse"`.
- The best checkpoint path is written to `best_ckpt_path.txt`.

---

## GP ONNX export (`prep_gp_ckpt.py`)

The GP export is more complex than the NN case due to GPyTorch’s lazy operators and data‑dependent control flow. The script implements a **two‑stage approach**:

1. Use GPyTorch to fully reconstruct the trained Exact GP and do a reference PyTorch inference.
2. Export a **custom ONNX‑friendly wrapper** that:
   - Uses the exact learned hyperparameters and training data,
   - Reimplements the predictive mean in pure tensor math (no GPyTorch in the graph).

### 1. Reconstructing the trained GP

- Reads `best_ckpt_path.txt` and loads the checkpoint on CPU.
- Verifies it’s a GP checkpoint (looks for `"gp."` keys); bails out if it looks like an NN checkpoint.
- Extracts:
  - `train_x`, `train_y` (scaled training inputs and targets),
  - `gp_state` (keys with `"gp."` prefix),
  - `mll_state` (keys with `"mll."` prefix).
- Rebuilds:
  - `likelihood = GaussianLikelihood()`,
  - `gp = ExactResidualGP(train_x, train_y, likelihood)` with a Matern‑5/2 ARD kernel,
  - Loads `gp_state` into `gp` and `mll_state` into a new `ExactMarginalLogLikelihood`.
- Loads `scaler_x` and `scaler_y` from `scalers/*.joblib` if available.
- Loads 5 realistic examples from `../../frontend/data/concrete.json`, `mortar.json`, `rock.json`, builds feature vectors in `FEATURE_COLS` order, and optionally scales them via `scaler_x`.
- Runs **reference PyTorch inference**:
  - With `fast_pred_var()` and `no_grad`, gets `output_dist = likelihood(gp(input_tensor))`.
  - Uses `output_dist.mean` as the scaled residual mean.
  - Inverse‑transforms via `scaler_y` (if present) to get residuals in MPa.
  - Computes base physics prediction for each example and prints:
    - Residual, baseline, final predicted strength, and true strength.

At this point, `prep_gp_ckpt.py` has a trusted **reference GP** in PyTorch to compare against.

### 2. ONNX‑friendly GP wrapper (`GPOnnxWrapper`)

Because ONNX cannot handle GPyTorch’s lazy operators and some data‑dependent logic, the script exports a custom wrapper:

- **Class**: `GPOnnxWrapper(nn.Module)`
- **Constructor**: `__init__(gp, likelihood, train_x, train_y)`
  - Detaches and stores `train_x` as a buffer (`(n_train, D)`).
  - Extracts learned hyperparameters from the trained GP:
    - ARD lengthscales: `gp.covar_module.base_kernel.lengthscale` → `(D,)`
    - Output scale: `gp.covar_module.outputscale` → scalar
    - Noise variance: `likelihood.noise` → scalar
  - Precomputes the **training covariance matrix**:
    - `K_xx = gp.covar_module(train_x, train_x).evaluate().float()`
    - Adds noise and a small jitter:
      - `K_xx_noisy = K_xx + (noise + 1e-6) * I`
  - Solves for:
    - `alpha = K_xx_noisy^{-1} train_y` in scaled space and registers it as a buffer.

- **Kernel implementation**: `_matern52_kernel(X, Z)`
  - Implements a Matern‑5/2 ARD kernel in pure tensor math:
    - Scales each dimension by the ARD lengthscales,
    - Computes pairwise distances and applies the standard Matern 5/2 formula,
    - Multiplies result by the learned output scale.
  - Diagnostics in the script show that this matches `gp.covar_module(train_x, input_tensor).evaluate()` to within ~1e‑6.

- **Forward**: `forward(x)`
  - Given test inputs `x` (already scaled), computes:
    - `K_xs = _matern52_kernel(train_x, x)` → `(n_train, batch)`
    - `mean = (K_xs^T @ alpha)` → `(batch, 1)` scaled residual mean.
  - Returns `(batch, 1)` scaled residuals, matching the GP’s predictive mean.

This module is small, deterministic, and uses only standard tensor operations, making it safe for ONNX export.

### ONNX export and verification

- **Export**:
  - Uses `torch.onnx.export` with:
    - `model = onnx_model` (the `GPOnnxWrapper`),
    - `dummy_input = torch.randn(1, input_dim)`,
    - `opset_version=14`,
    - Dynamic batch dimension on input and output,
    - `dynamo=False` to force the legacy TorchScript-based ONNX exporter (avoids `torch.export` constraints).
  - Writes `physics_gp.onnx`.

- **Diagnostics (PyTorch side)**:
  - Compares kernels: `gp.covar_module(train_x, input_tensor)` vs `_matern52_kernel(train_x, input_tensor)`.
  - Compares predictive means (scaled space): `likelihood(gp(input_tensor)).mean` vs `onnx_model(input_tensor)`.
  - Both match within ~1e‑6.

- **ONNX verification**:
  - Loads `physics_gp.onnx` with `onnxruntime`.
  - Runs the same scaled inputs through ONNX.
  - If `scaler_y` is available, inverse‑transforms outputs and compares to the GPyTorch residuals:
    - Prints per‑example PyTorch vs ONNX residual and absolute differences,
    - Prints the max difference; in the current implementation this is on the order of `1e-5` MPa.
  - **Performance benchmark**:
    - Runs 1000 iterations of:
      - PyTorch `GPOnnxWrapper` on the scaled batch,
      - ONNX Runtime on the same batch.
    - Prints per‑batch latency and speedup (typically ~4–5× faster with ONNX Runtime on CPU).

- **Standalone production example**:
  - Mirrors the NN example:
    - Demonstrates preparing unscaled features,
    - Scaling with `scaler_x`,
    - Running ONNX,
    - Inverse‑transforming with `scaler_y`,
    - Adding the physics baseline to get final compressive strength.

### GP ONNX caveats and guarantees

- **Only the mean is exported**:
  - The ONNX model returns only the **predictive mean of the residual** in scaled space.
  - Predictive variances / covariances are not exported.

- **Exact mean, approximate internals**:
  - While the runtime graph is a hand‑written kernel, it is parameterized directly by:
    - The **exact learned ARD lengthscales, outputscale, and noise** from the trained GPyTorch model,
    - The **exact training inputs and targets** (`train_x`, `train_y`) used to compute `alpha`.
  - Diagnostics show that the predictive mean from ONNX matches the GPyTorch ExactGP predictive mean within numerical precision for the tested inputs.

- **Runtime dependencies**:
  - **Export-time**: requires PyTorch, GPyTorch, and `linear_operator` to rebuild the trained GP and compute `alpha`.
  - **Inference-time**: only requires `onnxruntime` and the scalers (`scaler_x`, `scaler_y`) on the host side; the ONNX graph itself contains only vanilla tensor operations.

---

## Running the scripts

- **Train NN model and export ONNX**:

  ```bash
  cd backend/scripts
  make train_nn        # trains PhysicsNN, saves best checkpoint + best_ckpt_path.txt
  make prep_nn         # runs prep_nn_ckpt.py, exports physics_nn.onnx and verifies it
  ```

- **Train GP model and export ONNX**:

  ```bash
  cd backend/scripts
  make train_gp        # trains PhysicsGPR, saves best checkpoint + best_ckpt_path.txt
  make prep_gp         # runs prep_gp_ckpt.py, exports physics_gp.onnx and verifies it
  ```

- **Clean artifacts**:

  ```bash
  cd backend/scripts
  make clean           # removes checkpoints, logs, and best_ckpt_path.txt
  ```

In all cases, the ONNX models (`physics_nn.onnx`, `physics_gp.onnx`) are **residual predictors only**. Production systems are expected to:

1. Construct feature vectors in `FEATURE_COLS` order.
2. Scale with `scaler_x`.
3. Run the appropriate ONNX model to get a **scaled residual**.
4. Inverse‑transform with `scaler_y` to get the residual in MPa.
5. Compute the physics baseline via `base_physics_model`.
6. Add baseline + residual to obtain the final compressive strength prediction.
