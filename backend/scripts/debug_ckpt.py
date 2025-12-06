import time

import pandas as pd
import torch

from lightning_dataloader import (
    load_df,
    load_scalers,
    make_dataloaders_with_split,
)
from physics_gp import PhysicsGPR
from physics_nn import PhysicsNN
from physics_shared import combine_physics_with_residual

with open("best_ckpt_path.txt", "r") as f:
    ckpt_path = f.read()

df = load_df("../../frontend/data")
dl_train, dl_val, df_proc, _, _, idx_train, idx_val = make_dataloaders_with_split(
    df, val_size=0.2, seed=42
)

# Load the scalers that were created during training
sx, sy = load_scalers()

# Determine which model to load by inspecting the checkpoint
ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
is_gp = any(k.startswith("gp.") for k in ckpt["state_dict"].keys())

if is_gp:
    model_class = PhysicsGPR
    print("Loading GP model")
else:
    model_class = PhysicsNN
    print("Loading NN model")

model = model_class.load_from_checkpoint(ckpt_path, scaler_x=sx, scaler_y=sy)

model.eval()
# get validation samples: dl_val is a DataLoader
start_time = time.time()
preds = combine_physics_with_residual(df_proc, model)
end_time = time.time()
inference_time = end_time - start_time
avg_time = inference_time / df_proc.shape[0]
print(f"Average time per sample: {avg_time} seconds")
print(preds)

preds_with_truth = pd.concat([df_proc, preds.drop(columns=["residual"])], axis=1)
pred_col = "pred_mpa"
preds_with_truth["error"] = (
    preds_with_truth[pred_col] - preds_with_truth["y_strength_mpa"]
)
summary_cols = [pred_col, "pred_lo_mpa", "pred_hi_mpa", "y_strength_mpa", "error"]
print(preds_with_truth[summary_cols])

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.scatter(
    preds_with_truth["y_strength_mpa"].iloc[idx_train],
    preds_with_truth[pred_col].iloc[idx_train],
    c="black",
    alpha=0.5 / 4,
    s=80,
    label="Training samples",
)
plt.scatter(
    preds_with_truth["y_strength_mpa"].iloc[idx_val],
    preds_with_truth[pred_col].iloc[idx_val],
    c="red",
    alpha=0.5,
    s=150,
    label="Validation samples",
)
# scatter the true values against the residuals for all samples, plot in xkcd:forest-green
# plt.scatter(
#     preds_with_truth["y_strength_mpa"],
#     preds_with_truth["residual"],
#     c="xkcd:forest green",
#     alpha=0.5,
#     s=10,
# )

# draw horizontal dashed lines from (y_true, y_true) to (y_true, y_pred) between true and predicted values for idx_val
plt.vlines(
    x=preds_with_truth["y_strength_mpa"].iloc[idx_val],
    ymin=preds_with_truth["y_strength_mpa"].iloc[idx_val],
    ymax=preds_with_truth[pred_col].iloc[idx_val],
    color="black",
    linestyle="--",
)
plt.xlabel("True Strength (MPa)")
plt.ylabel("Predicted Strength (MPa)")
plt.title("True vs Predicted Strength")
# aspect ratio 1:1
# plt.gca().set_aspect(0.5, adjustable='box')
plt.legend()
plt.savefig("true_vs_predicted.png")
# plt.show()

import gpytorch

# Print kernel hyperparameters
if hasattr(model, "gp"):
    if hasattr(model.gp.covar_module, "base_kernel"):
        # Check if lengthscale is there (typical for ARD kernels)
        if hasattr(model.gp.covar_module.base_kernel, "lengthscale"):
            print("Lengthscale:", model.gp.covar_module.base_kernel.lengthscale)
        print("Kernel parameters:", model.gp.covar_module.base_kernel.hyperparameters())

# Complete model state
print("Model state dict:")
for name, param in model.named_parameters():
    print(f"{name}: {param}")

# Explanation to the user
print("Explanation:")
print(
    "- If the model was trained with synthetic data using the Hirsch model, we expect feature importances \n"
    "  to show little variance since irrelevant features were not contributing."
)
print(
    "- The lengthscale values should be roughly similar across features, indicating no particular \n"
    "  feature is overly dominant. This means the model isn't wrongfully attributing significance\n"
    "  to any irrelevant synthetics."
)
print(
    "- Low noise and consistent parameters are good indications, showing the model fits well to the \n"
    "  baseline predictions. Any significant deviation might suggest strong influences on the model \n"
    "  from noise or artifacts not present."
)

print(
    "the GPR is acting as a 'residual tuner.' If the initial physics-based prediction is close, it works beautifully."
    "If the initial prediction is far off, the GPR doesn't have enough information or capacity to learn the "
    "correction from scratch, and the training process fails."
)
