import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from gp_lightning_dataloader import (
    PhysicsGPR,
    combine_hirsch_with_residual,
    load_df,
    make_dataloaders_with_split,
)

df = load_df("../../data")
dl_train, dl_val, df_proc, sx, sy, _, idx_val = make_dataloaders_with_split(
    df, val_size=0.2, seed=42
)


model = PhysicsGPR(sx, sy)

ckpt = ModelCheckpoint(
    dirpath="checkpoints",
    filename="phys-gp-{epoch:03d}-{val_rmse:.4f}",
    monitor="val_rmse",
    mode="min",
    save_top_k=1,
)

trainer = L.Trainer(
    max_epochs=200,
    callbacks=[ckpt],
    log_every_n_steps=1,
)
trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)
best_ckpt_path = ckpt.best_model_path
print("Best ckpt:", best_ckpt_path)
with open("best_ckpt_path.txt", "w") as f:
    f.write(best_ckpt_path)

# Load best and run inference
loaded = PhysicsGPR.load_from_checkpoint(best_ckpt_path, scaler_x=sx, scaler_y=sy)
loaded.eval()
preds = combine_hirsch_with_residual(df_proc.iloc[idx_val].head(5), loaded)
print("Sample predictions:")
print(preds)
