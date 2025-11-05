import argparse

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from gp_lightning_dataloader import (
    FEATURE_COLS,
    PhysicsGPR,
    PhysicsNN,
    combine_hirsch_with_residual,
    load_df,
    make_dataloaders_with_split,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="nn", choices=["gp", "nn"])
args = parser.parse_args()

df = load_df("../../data")
dl_train, dl_val, df_proc, sx, sy, _, idx_val = make_dataloaders_with_split(
    df, val_size=0.2, seed=42
)


if args.model == "gp":
    model = PhysicsGPR(sx, sy)
    filename_template = "phys-gp-{epoch:03d}-{val_rmse:.4f}"
else:
    model = PhysicsNN(input_dim=len(FEATURE_COLS), scaler_x=sx, scaler_y=sy)
    filename_template = "phys-nn-{epoch:03d}-{val_rmse:.4f}"


ckpt = ModelCheckpoint(
    dirpath="checkpoints",
    filename=filename_template,
    monitor="val_rmse",
    mode="min",
    save_top_k=1,
)

trainer = L.Trainer(
    max_epochs=200,
    callbacks=[ckpt],
    log_every_n_steps=10,
)
trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)
best_ckpt_path = ckpt.best_model_path
print("Best ckpt:", best_ckpt_path)
with open("best_ckpt_path.txt", "w") as f:
    f.write(best_ckpt_path)

# Load best and run inference
if args.model == "gp":
    loaded = PhysicsGPR.load_from_checkpoint(best_ckpt_path, scaler_x=sx, scaler_y=sy)
else:
    loaded = PhysicsNN.load_from_checkpoint(best_ckpt_path, scaler_x=sx, scaler_y=sy)

loaded.eval()
preds = combine_hirsch_with_residual(df_proc.iloc[idx_val].head(5), loaded)
print("Sample predictions:")
print(preds)
