import time

import pandas as pd
import torch

from gp_lightning_dataloader import (
    FEATURE_COLS,
    PhysicsGPR,
    combine_hirsch_with_residual,
    load_df,
    make_dataloaders_with_split,
)

with open("best_ckpt_path.txt", "r") as f:
    ckpt_path = f.read()

df = load_df("../../data")
dl_train, dl_val, df_proc, sx, sy, idx_train, idx_val = make_dataloaders_with_split(
    df, val_size=0.2, seed=42
)

model = PhysicsGPR.load_from_checkpoint(ckpt_path, scaler_x=sx, scaler_y=sy)
model.eval()
# get validation samples: dl_val is a DataLoader
start_time = time.time()
preds = combine_hirsch_with_residual(df_proc, model)
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
)
plt.scatter(
    preds_with_truth["y_strength_mpa"].iloc[idx_val],
    preds_with_truth[pred_col].iloc[idx_val],
    c="red",
    alpha=0.5,
    s=150,
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
plt.savefig("true_vs_predicted.png")
plt.show()
