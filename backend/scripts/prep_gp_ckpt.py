# prep_gp_ckpt.py
import json
import math
from pathlib import Path

import gpytorch
import joblib
import numpy as np
import torch
import torch.nn as nn

from physics_gp import ExactResidualGP
from physics_shared import base_physics_model

# Load checkpoint path
try:
    with open("best_ckpt_path.txt", "r") as f:
        ckpt_path = f.read().strip()
except FileNotFoundError:
    # Fallback if running without training first, but typically expected
    print("best_ckpt_path.txt not found. Please run training first.")
    exit(1)

print(f"Loading checkpoint from: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)

# Check if it's a GP model
is_gp = any(k.startswith("gp.") for k in ckpt.keys()) or any(
    k.startswith("gp.") for k in ckpt.get("state_dict", {}).keys()
)

if not is_gp:
    # It might be that the keys are not prefixed if saved differently,
    # but based on physics_gp.py they should be.
    # If it's an NN checkpoint, we should warn.
    # NN checkpoint has "model." keys usually.
    is_nn = any(k.startswith("model.") for k in ckpt.get("state_dict", {}).keys())
    if is_nn:
        raise ValueError(
            "This script is for GP models only. Detected NN model in checkpoint."
        )

# Extract training data needed for ExactGP
if "train_x" not in ckpt or "train_y" not in ckpt:
    raise ValueError(
        "Checkpoint does not contain train_x/train_y needed for GP initialization."
    )

train_x = ckpt["train_x"]
train_y = ckpt["train_y"]
print(f"Found training data in checkpoint: X={train_x.shape}, y={train_y.shape}")

# Initialize GP model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp = ExactResidualGP(train_x, train_y, likelihood)

# Load GP state
# PhysicsGPR saves gp state with "gp." prefix in the root ckpt dict
gp_state = {}
mll_state = {}

# Check root keys first (as per physics_gp.py on_save_checkpoint)
for k, v in ckpt.items():
    if k.startswith("gp."):
        gp_state[k[3:]] = v
    elif k.startswith("mll."):
        mll_state[k[4:]] = v

# Load state dicts
if gp_state:
    gp.load_state_dict(gp_state)
    print(f"✓ Loaded GP state ({len(gp_state)} keys)")

# Load MLL/Likelihood state
# MLL state contains likelihood parameters
if mll_state:
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
    mll.load_state_dict(mll_state)
    print(f"✓ Loaded MLL/Likelihood state ({len(mll_state)} keys)")
else:
    print(
        "⚠ Warning: No MLL state found. Likelihood parameters (noise) might be uninitialized."
    )

gp.eval()
likelihood.eval()

print("\n✓ Successfully loaded GP model")

# Load scalers
try:
    scaler_x = joblib.load("scalers/scaler_x.joblib")
    scaler_y = joblib.load("scalers/scaler_y.joblib")
    print("✓ Loaded scalers")
except FileNotFoundError:
    print("⚠ Warning: Scalers not found. Using unscaled inputs/outputs.")
    scaler_x = None
    scaler_y = None

# Load realistic data
data_dir = Path("../../frontend/data")
concrete_data = json.loads((data_dir / "concrete.json").read_text())
mortar_data = json.loads((data_dir / "mortar.json").read_text())
rock_data = json.loads((data_dir / "rock.json").read_text())

FEATURE_COLS = [
    "mortar_splitting_strength_mpa",
    "mortar_shrinkage_in",
    "mortar_flexural_strength_mpa",
    "mortar_slump_in",
    "mortar_compressive_strength_mpa",
    "mortar_poissons_ratio",
    "rock_compressive_strength_mpa",
    "rock_size_in",
    "rock_density_lb_ft3",
    "rock_specific_gravity",
    "rock_ratio",
]

# Use a small but slightly richer set of realistic examples
sample_keys = list(concrete_data.keys())[:5]
print(f"\nUsing {len(sample_keys)} realistic examples from data:")
print("=" * 80)

realistic_inputs = []
for key in sample_keys:
    rec = concrete_data[key]
    mortar_id = rec["mortar_id"]
    rock_id = rec["rock_id"]
    mortar_props = mortar_data[mortar_id]["properties"]
    rock_props = rock_data[rock_id]["properties"]

    features = [
        mortar_props["splitting_strength_mpa"],
        mortar_props["shrinkage_inches"],
        mortar_props["flexural_strength_mpa"],
        mortar_props["slump_inches"],
        mortar_props["compressive_strength_mpa"],
        mortar_props["poissons_ratio"],
        rock_props["compressive_strength_mpa"],
        rock_props["size_inches"],
        rock_props["density_lb_ft3"],
        rock_props["specific_gravity"],
        rec["rock_ratio"],
    ]
    realistic_inputs.append(features)

    # Store for later
    rec["mortar_props"] = mortar_props
    rec["rock_props"] = rock_props

    print(f"\nExample: {key}")
    print(
        f"  Features: {[f'{f:.3f}' if isinstance(f, float) else f for f in features]}"
    )

realistic_inputs = np.array(realistic_inputs, dtype=np.float32)

if scaler_x is not None:
    realistic_inputs_scaled = scaler_x.transform(realistic_inputs)
else:
    realistic_inputs_scaled = realistic_inputs

input_tensor = torch.tensor(realistic_inputs_scaled, dtype=torch.float32)

print(f"\n{'='*80}")
print("Running inference (PyTorch)...")

# Run inference
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Predict mean residual
    output_dist = likelihood(gp(input_tensor))
    output_scaled = output_dist.mean

if scaler_y is not None:
    output_numpy = output_scaled.cpu().numpy()
    output_unscaled = scaler_y.inverse_transform(output_numpy.reshape(-1, 1)).ravel()
else:
    output_unscaled = output_scaled.cpu().numpy().ravel()

print(f"\nResults:")
print(f"{'Example':<20} {'Predicted Residual':<25} {'Predicted Strength*':<25}")
print("-" * 80)

for i, key in enumerate(sample_keys):
    rec = concrete_data[key]
    predicted_residual = output_unscaled[i]

    # Base physics
    mortar_comp = rec["mortar_props"]["compressive_strength_mpa"]
    rock_comp = rec["rock_props"]["compressive_strength_mpa"]
    rock_ratio = rec["rock_ratio"]

    base_pred = base_physics_model(
        {
            "mortar_compressive_strength_mpa": mortar_comp,
            "rock_compressive_strength_mpa": rock_comp,
            "rock_ratio": rock_ratio,
        }
    )
    final_pred = base_pred + predicted_residual
    true_strength = rec["concrete_compressive_strength_mpa"]

    print(
        f"{key:<20} {predicted_residual:>10.2f} MPa (residual) {base_pred:>10.2f} (base) {final_pred:>10.2f} (final) | True: {true_strength:.2f}"
    )

# ============================================================================
# ONNX Export and Testing
# ============================================================================
print("\n" + "=" * 80)
print("ONNX Export and Testing")
print("=" * 80)

onnx_path = "physics_gp.onnx"


# Wrapper for ONNX export
# NOTE:
# - We avoid calling into gpytorch / linear_operator in forward(), since those
#   APIs contain data-dependent control flow that is not ONNX-export-friendly.
# - Instead, we implement the Exact GP predictive mean with a Matern-5/2 kernel
#   directly in PyTorch using the learned hyperparameters from the trained GP.
class GPOnnxWrapper(nn.Module):
    def __init__(self, gp, likelihood, train_x, train_y):
        super().__init__()
        # Detach training data and register as buffers so they become constants
        train_x = train_x.detach().float()
        train_y = train_y.detach().float()
        self.register_buffer("train_x", train_x)

        # Extract hyperparameters from trained gpytorch model
        lengthscale = (
            gp.covar_module.base_kernel.lengthscale.detach().view(-1).float()
        )  # (D,)
        outputscale = gp.covar_module.outputscale.detach().view([]).float()  # scalar
        noise = likelihood.noise.detach().view([]).float()  # scalar noise variance

        self.register_buffer("lengthscale", lengthscale)
        self.register_buffer("outputscale", outputscale)
        self.register_buffer("noise", noise)

        # Precompute K_xx using the exact gpytorch kernel for training inputs
        with torch.no_grad():
            K_xx = gp.covar_module(train_x, train_x).evaluate().float()

        # Precompute alpha = (K_xx + sigma_n^2 I)^{-1} y  (in scaled residual space)
        alpha = self._compute_alpha(K_xx, train_y)
        self.register_buffer("alpha", alpha)

    def _matern52_kernel(self, X, Z):
        """
        Matern 5/2 kernel with ARD lengthscales.
        X: (n, D), Z: (m, D)
        Returns: (n, m) covariance matrix.
        """
        ls = self.lengthscale  # (D,)
        Xs = X / ls
        Zs = Z / ls

        X2 = (Xs**2).sum(dim=1, keepdim=True)  # (n, 1)
        Z2 = (Zs**2).sum(dim=1, keepdim=True)  # (m, 1)
        cross = Xs @ Zs.T  # (n, m)

        d2 = X2 + Z2.T - 2.0 * cross  # (n, m)
        d2 = torch.clamp(d2, min=0.0)
        r = torch.sqrt(d2 + 1e-9)

        sqrt5 = math.sqrt(5.0)
        k = (1.0 + sqrt5 * r + 5.0 * r * r / 3.0) * torch.exp(-sqrt5 * r)
        return self.outputscale * k

    def _compute_alpha(self, K_xx, train_y):
        # K_xx + sigma_n^2 I
        n = K_xx.shape[0]
        K_xx = K_xx + (self.noise + 1e-6) * torch.eye(
            n, device=K_xx.device, dtype=K_xx.dtype
        )

        # ZeroMean in PhysicsGPR, so no need to subtract a mean term
        alpha = torch.linalg.solve(K_xx, train_y.unsqueeze(-1)).squeeze(-1)
        return alpha

    def forward(self, x):
        # Predictive mean: K_xs^T alpha
        # x: (batch, D)
        K_xs = self._matern52_kernel(self.train_x, x)  # (n_train, batch)
        alpha = self.alpha.unsqueeze(-1)  # (n_train, 1)
        mean = K_xs.transpose(0, 1) @ alpha  # (batch, 1)
        return mean  # (batch, 1) to match previous shape


onnx_model = GPOnnxWrapper(gp, likelihood, train_x, train_y)
onnx_model.eval()

# --------------------------------------------------------------------------
# Diagnostic: compare our hand-written Matern 5/2 kernel against
# GPyTorch's Matern kernel on the current realistic inputs.
# This helps ensure K(xs, X) matches numerically.
# --------------------------------------------------------------------------
with torch.no_grad():
    gp_K_xs = gp.covar_module(train_x, input_tensor).evaluate().float()
    our_K_xs = onnx_model._matern52_kernel(train_x, input_tensor)
    diff_K = (gp_K_xs - our_K_xs).abs()
    print("\nKernel diagnostics (train_x vs realistic inputs):")
    print(f"  gp_K_xs shape:   {gp_K_xs.shape}")
    print(f"  our_K_xs shape:  {our_K_xs.shape}")
    print(f"  max |ΔK_xs|:     {diff_K.max().item():.6e}")
    print(f"  mean |ΔK_xs|:    {diff_K.mean().item():.6e}")

    # Also compare predictive means in scaled space
    wrapper_scaled = onnx_model(input_tensor).squeeze(-1)  # (N,)
    gp_scaled = output_scaled.detach().squeeze(-1)
    mean_diff = (wrapper_scaled - gp_scaled).abs()
    print("\nPredictive mean diagnostics (scaled space):")
    print(f"  max |Δmean|:     {mean_diff.max().item():.6e}")
    print(f"  mean |Δmean|:    {mean_diff.mean().item():.6e}")

sample_input = torch.randn(1, train_x.shape[1])

print(f"\nExporting model to ONNX format...")
# Note: gpytorch models are dynamic by default but we can export the mean calculation.
# Use the legacy (non-dynamo) exporter to avoid torch.export / FakeTensor
torch.onnx.export(
    onnx_model,
    sample_input,
    onnx_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    dynamo=False,
)
print(f"✓ Model exported to {onnx_path}")

# Test with ONNX Runtime
try:
    import time

    import onnxruntime as ort

    print(f"\nLoading ONNX model with ONNX Runtime...")
    ort_session = ort.InferenceSession(onnx_path)
    print(f"✓ ONNX model loaded")

    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    print(f"  Input name: {input_name}")
    print(f"  Output name: {output_name}")

    print(f"\nRunning ONNX inference on same test data...")
    onnx_input = realistic_inputs_scaled.astype(np.float32)

    # Single pass for correctness comparison
    onnx_output_scaled = ort_session.run([output_name], {input_name: onnx_input})[0]

    # Unscale
    if scaler_y is not None:
        onnx_output_unscaled = scaler_y.inverse_transform(
            onnx_output_scaled.reshape(-1, 1)
        ).ravel()
    else:
        onnx_output_unscaled = onnx_output_scaled.ravel()

    # Comparison
    print(f"\n{'='*80}")
    print("PyTorch vs ONNX Comparison:")
    print(f"{'Example':<20} {'PyTorch':<15} {'ONNX':<15} {'Diff':<15}")
    print("-" * 80)

    max_diff = 0.0
    for i, key in enumerate(sample_keys):
        pytorch_pred = output_unscaled[i]
        onnx_pred = onnx_output_unscaled[i]
        diff = abs(pytorch_pred - onnx_pred)
        max_diff = max(max_diff, diff)
        print(f"{key:<20} {pytorch_pred:>14.4f} {onnx_pred:>14.4f} {diff:>14.6f}")

    print(f"\nMax difference: {max_diff:.8f} MPa")
    if max_diff < 1e-4:
        print("✓ ONNX outputs match PyTorch")
    else:
        print("⚠ Warning: Differences found between PyTorch and ONNX")

    # ------------------------------------------------------------------
    # Performance Benchmark (fair production-style comparison)
    # We compare:
    #   - PyTorch running the exported GPOnnxWrapper (eager mode)
    #   - ONNX Runtime running the exported physics_gp.onnx
    # This mirrors how inference would run in production for each stack.
    # ------------------------------------------------------------------
    n_iterations = 1000
    print(f"\n{'='*80}")
    print("Performance Benchmark ({} iterations):".format(n_iterations))
    print("-" * 80)

    # PyTorch benchmark (GPOnnxWrapper)
    torch_input = torch.tensor(realistic_inputs_scaled, dtype=torch.float32)
    start = time.time()
    for _ in range(n_iterations):
        with torch.no_grad():
            _ = onnx_model(torch_input)
    torch_time = (time.time() - start) / n_iterations * 1000  # ms per batch

    # ONNX Runtime benchmark
    start = time.time()
    for _ in range(n_iterations):
        _ = ort_session.run([output_name], {input_name: onnx_input})
    onnx_time = (time.time() - start) / n_iterations * 1000  # ms per batch

    print(
        f"PyTorch (GPOnnxWrapper): {torch_time:.4f} ms per batch ({len(sample_keys)} samples)"
    )
    print(
        f"ONNX Runtime:           {onnx_time:.4f} ms per batch ({len(sample_keys)} samples)"
    )
    if onnx_time > 0:
        print(f"Speedup:                {torch_time/onnx_time:.2f}x")

    print("\n✓ ONNX export and testing successful!")
    print(f"✓ Model ready for production deployment at: {onnx_path}")

except ImportError:
    print("\n⚠ onnxruntime not installed. Install with: pip install onnxruntime")
    print(f"  ONNX model exported to {onnx_path} but not tested.")
except Exception as e:
    print(f"\n✗ Error during ONNX testing: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# Standalone ONNX Inference Example (Mock Data)
# ============================================================================
print("\n" + "=" * 80)
print("Standalone ONNX Inference Example (Mock Data)")
print("=" * 80)

try:
    import time

    import onnxruntime as ort

    # 1. Load the model
    print(f"Loading {onnx_path}...")
    session = ort.InferenceSession(onnx_path)

    # 2. Prepare mock input
    # Shape: (batch_size, input_dim)
    batch_size = 2
    # input_dim is inferred from train_x shape
    input_dim = train_x.shape[1]
    mock_input = np.random.randn(batch_size, input_dim).astype(np.float32)

    print(f"\nMock Input (shape {mock_input.shape}):")
    print(mock_input)

    # 3. Run Inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    start_time = time.time()
    result = session.run([output_name], {input_name: mock_input})[0]
    latency = (time.time() - start_time) * 1000

    print(f"\nInference Output (shape {result.shape}):")
    print(result)
    print(f"Latency: {latency:.3f} ms")

    print("\nNOTE: This output is the scaled residual. In production:")
    print("  1. Scale input features (scaler_x)")
    print("  2. Run ONNX inference")
    print("  3. Inverse transform output (scaler_y) -> predicted_residual")
    print("  4. Final Prediction = Base_Physics_Model + Predicted_Residual")

    # ============================================================================
    # Demonstrate Production Inference Steps
    # ============================================================================
    print("\n" + "=" * 80)
    print("Demonstrating Production Inference Steps")
    print("=" * 80)

    if scaler_x is None or scaler_y is None:
        print("⚠ Skipping production demo - scalers not available")
    else:
        # Step 1: Create realistic mock input (unscaled features)
        print("\nStep 1: Prepare realistic input features (unscaled)")
        print("-" * 80)

        # Create a mock example with realistic values
        mock_features_unscaled = np.array(
            [
                [
                    3.2,  # mortar_splitting_strength_mpa
                    0.042,  # mortar_shrinkage_in
                    4.8,  # mortar_flexural_strength_mpa
                    4.1,  # mortar_slump_in
                    28.5,  # mortar_compressive_strength_mpa
                    0.18,  # mortar_poissons_ratio
                    200.0,  # rock_compressive_strength_mpa
                    0.75,  # rock_size_in
                    165.4,  # rock_density_lb_ft3
                    2.65,  # rock_specific_gravity
                    0.4,  # rock_ratio
                ],
                [
                    4.1,  # mortar_splitting_strength_mpa
                    0.038,  # mortar_shrinkage_in
                    5.2,  # mortar_flexural_strength_mpa
                    3.8,  # mortar_slump_in
                    35.0,  # mortar_compressive_strength_mpa
                    0.20,  # mortar_poissons_ratio
                    180.0,  # rock_compressive_strength_mpa
                    0.5,  # rock_size_in
                    160.0,  # rock_density_lb_ft3
                    2.60,  # rock_specific_gravity
                    0.35,  # rock_ratio
                ],
            ],
            dtype=np.float32,
        )

        print(f"Input shape: {mock_features_unscaled.shape}")
        print("Sample features (first row):")
        for i, col in enumerate(FEATURE_COLS):
            print(f"  {col}: {mock_features_unscaled[0, i]:.3f}")

        # Step 2: Scale input features
        print("\nStep 2: Scale input features using scaler_x")
        print("-" * 80)
        mock_features_scaled = scaler_x.transform(mock_features_unscaled)
        print(f"Scaled input shape: {mock_features_scaled.shape}")
        print(f"Scaled values (first row, first 3): {mock_features_scaled[0, :3]}")

        # Step 3: Run ONNX inference
        print("\nStep 3: Run ONNX inference")
        print("-" * 80)
        start_time = time.time()
        onnx_output_scaled = session.run(
            [output_name], {input_name: mock_features_scaled.astype(np.float32)}
        )[0]
        inference_time = (time.time() - start_time) * 1000
        print(f"Inference output (scaled residual): {onnx_output_scaled.ravel()}")
        print(f"Inference latency: {inference_time:.3f} ms")

        # Step 4: Inverse transform output
        print("\nStep 4: Inverse transform output using scaler_y")
        print("-" * 80)
        predicted_residual = scaler_y.inverse_transform(
            onnx_output_scaled.reshape(-1, 1)
        ).ravel()
        print(f"Predicted residual (unscaled): {predicted_residual} MPa")

        # Step 5: Calculate final prediction
        print("\nStep 5: Calculate final prediction = Base Physics + Residual")
        print("-" * 80)
        for i in range(len(mock_features_unscaled)):
            mortar_comp = mock_features_unscaled[
                i, 4
            ]  # mortar_compressive_strength_mpa
            rock_comp = mock_features_unscaled[i, 6]  # rock_compressive_strength_mpa
            rock_ratio = mock_features_unscaled[i, 10]  # rock_ratio

            base_pred = base_physics_model(
                {
                    "mortar_compressive_strength_mpa": mortar_comp,
                    "rock_compressive_strength_mpa": rock_comp,
                    "rock_ratio": rock_ratio,
                }
            )
            final_pred = base_pred + predicted_residual[i]

            print(f"\nExample {i+1}:")
            print(f"  Base Physics Prediction: {base_pred:.2f} MPa")
            print(f"  Predicted Residual:      {predicted_residual[i]:.2f} MPa")
            print(f"  Final Prediction:        {final_pred:.2f} MPa")
            print(
                f"  (Base + Residual = {base_pred:.2f} + {predicted_residual[i]:.2f})"
            )

        print("\n✓ Production inference pipeline demonstrated successfully!")

except ImportError:
    print("onnxruntime not installed.")
except Exception as e:
    print(f"Error in standalone example: {e}")
