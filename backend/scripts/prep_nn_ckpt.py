# prep_nn_ckpt.py
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

# Load checkpoint
with open("best_ckpt_path.txt", "r") as f:
    ckpt_path = f.read().strip()

print(f"Loading checkpoint from: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)

# Check if it's a GP model (skip if so)
is_gp = any(k.startswith("gp.") for k in ckpt["state_dict"].keys())
if is_gp:
    raise ValueError(
        "This script is for NN models only. Detected GP model in checkpoint."
    )

# Extract hyperparameters
hyperparams = ckpt.get("hyper_parameters", {})
input_dim = hyperparams.get("input_dim", 11)  # Default to 11 (len(FEATURE_COLS))
hidden_dim = hyperparams.get("hidden_dim", 64)
n_hidden_layers = hyperparams.get("n_hidden_layers", 4)

print(f"Model architecture:")
print(f"  input_dim: {input_dim}")
print(f"  hidden_dim: {hidden_dim}")
print(f"  n_hidden_layers: {n_hidden_layers}")

# Create vanilla Sequential model (matching PhysicsNN structure)
layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
for _ in range(n_hidden_layers):
    layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
layers.append(nn.Linear(hidden_dim, 1))

vanilla_model = nn.Sequential(*layers)

# Extract state_dict and filter for model.* keys, removing "model." prefix
vanilla_model_state_dict = {}
for k, v in ckpt["state_dict"].items():
    if k.startswith("model."):
        # Remove "model." prefix
        new_key = k[len("model.") :]
        vanilla_model_state_dict[new_key] = v
    # Skip other keys (like scaler_x, scaler_y, hyperparameters, etc.)

print(f"\nFound {len(vanilla_model_state_dict)} model parameters")
print("Parameter keys:", list(vanilla_model_state_dict.keys())[:5], "...")

# Load state_dict into vanilla model
vanilla_model.load_state_dict(vanilla_model_state_dict)
vanilla_model.eval()

print("\n✓ Successfully loaded model without PyTorch Lightning")

# Load scalers (needed for proper input scaling)
try:
    scaler_x = joblib.load("scalers/scaler_x.joblib")
    scaler_y = joblib.load("scalers/scaler_y.joblib")
    print("✓ Loaded scalers")
except FileNotFoundError:
    print("⚠ Warning: Scalers not found. Using unscaled inputs/outputs.")
    scaler_x = None
    scaler_y = None

# Load realistic data from data files
data_dir = Path("../../frontend/data")
concrete_data = json.loads((data_dir / "concrete.json").read_text())
mortar_data = json.loads((data_dir / "mortar.json").read_text())
rock_data = json.loads((data_dir / "rock.json").read_text())

# FEATURE_COLS order
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

# Pick a few real examples from the data
sample_keys = list(concrete_data.keys())[:3]  # First 3 examples
print(f"\nUsing {len(sample_keys)} realistic examples from data:")
print("=" * 80)

realistic_inputs = []
for key in sample_keys:
    rec = concrete_data[key]
    mortar_id = rec["mortar_id"]
    rock_id = rec["rock_id"]

    mortar_props = mortar_data[mortar_id]["properties"]
    rock_props = rock_data[rock_id]["properties"]

    # Build feature vector in FEATURE_COLS order
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

    print(f"\nExample: {key}")
    print(f"  Mortar: {mortar_id} ({mortar_data[mortar_id]['name']})")
    print(f"  Rock: {rock_id} ({rock_data[rock_id]['name']})")
    print(f"  Rock ratio: {rec['rock_ratio']}")
    print(f"  True strength: {rec['concrete_compressive_strength_mpa']:.2f} MPa")
    print(
        f"  Features: {[f'{f:.3f}' if isinstance(f, float) else f for f in features]}"
    )

# Convert to numpy array and then tensor
realistic_inputs = np.array(realistic_inputs, dtype=np.float32)

# Scale inputs if scaler is available
if scaler_x is not None:
    realistic_inputs_scaled = scaler_x.transform(realistic_inputs)
else:
    realistic_inputs_scaled = realistic_inputs
    print("\n⚠ Using unscaled inputs (no scaler found)")

input_tensor = torch.tensor(realistic_inputs_scaled, dtype=torch.float32)

print(f"\n{'='*80}")
print("Running inference...")
print(f"Input shape: {input_tensor.shape}")

# Run inference
with torch.no_grad():
    output_scaled = vanilla_model(input_tensor)

# Unscale output if scaler is available
if scaler_y is not None:
    output_numpy = output_scaled.cpu().numpy()
    output_unscaled = scaler_y.inverse_transform(output_numpy.reshape(-1, 1)).ravel()
else:
    output_unscaled = output_scaled.cpu().numpy().ravel()
    print("\n⚠ Using unscaled outputs (no scaler found)")

print(f"\n{'='*80}")
print("Results:")
print(f"{'Example':<20} {'Predicted Residual':<25} {'Predicted Strength*':<25}")
print("-" * 80)

for i, key in enumerate(sample_keys):
    rec = concrete_data[key]
    true_strength = rec["concrete_compressive_strength_mpa"]
    predicted_residual = output_unscaled[i]

    # Calculate predicted strength (residual + hirsch prediction)
    # For simplicity, we'll just show the residual here
    # In practice, you'd add: predicted_strength = hirsch_prediction + predicted_residual
    print(f"{key:<20} {predicted_residual:>24.2f} MPa (residual)")

print("\n* Note: Predicted strength = Hirsch prediction + residual")
print("  This script only shows the residual component.")
print("\n✓ Inference test successful with realistic data!")

# ============================================================================
# ONNX Export and Testing
# ============================================================================
print("\n" + "=" * 80)
print("ONNX Export and Testing")
print("=" * 80)

# Export model to ONNX
onnx_path = "physics_nn.onnx"
dummy_input = torch.randn(1, input_dim)

print(f"\nExporting model to ONNX format...")
torch.onnx.export(
    vanilla_model,
    dummy_input,
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
)
print(f"✓ Model exported to {onnx_path}")

# Test with ONNX Runtime
try:
    import onnxruntime as ort

    print(f"\nLoading ONNX model with ONNX Runtime...")
    ort_session = ort.InferenceSession(onnx_path)
    print(f"✓ ONNX model loaded")

    # Get model info
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    print(f"  Input name: {input_name}")
    print(f"  Output name: {output_name}")

    # Run inference with ONNX Runtime
    print(f"\nRunning ONNX inference on same test data...")
    onnx_input = realistic_inputs_scaled.astype(np.float32)
    onnx_output_scaled = ort_session.run([output_name], {input_name: onnx_input})[0]

    # Unscale ONNX output
    if scaler_y is not None:
        onnx_output_unscaled = scaler_y.inverse_transform(
            onnx_output_scaled.reshape(-1, 1)
        ).ravel()
    else:
        onnx_output_unscaled = onnx_output_scaled.ravel()

    # Compare PyTorch vs ONNX results
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

    if max_diff < 1e-5:
        print("✓ ONNX outputs match PyTorch (within numerical precision)")
    elif max_diff < 1e-3:
        print("✓ ONNX outputs match PyTorch (small numerical differences)")
    else:
        print("⚠ Warning: Significant differences between PyTorch and ONNX")

    # Benchmark inference speed
    import time

    n_iterations = 1000

    # PyTorch benchmark
    print(f"\n{'='*80}")
    print("Performance Benchmark ({} iterations):".format(n_iterations))
    print("-" * 80)

    torch_input = torch.tensor(realistic_inputs_scaled, dtype=torch.float32)
    start = time.time()
    for _ in range(n_iterations):
        with torch.no_grad():
            _ = vanilla_model(torch_input)
    torch_time = (time.time() - start) / n_iterations * 1000  # ms

    # ONNX benchmark
    start = time.time()
    for _ in range(n_iterations):
        _ = ort_session.run([output_name], {input_name: onnx_input})
    onnx_time = (time.time() - start) / n_iterations * 1000  # ms

    print(f"PyTorch:      {torch_time:.4f} ms per batch ({len(sample_keys)} samples)")
    print(f"ONNX Runtime: {onnx_time:.4f} ms per batch ({len(sample_keys)} samples)")
    print(f"Speedup:      {torch_time/onnx_time:.2f}x")

    print("\n✓ ONNX export and testing successful!")
    print(f"✓ Model ready for production deployment at: {onnx_path}")

except ImportError:
    print("\n⚠ onnxruntime not installed. Install with: pip install onnxruntime")
    print(f"  ONNX model exported to {onnx_path} but not tested.")
except Exception as e:
    print(f"\n✗ Error during ONNX testing: {e}")
    import traceback

    traceback.print_exc()
