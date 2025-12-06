# gp_lightning_dataloader.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from physics_shared import FEATURE_COLS, base_physics_model


def make_dataloaders_with_split(
    df: pd.DataFrame, val_size=0.2, seed=42, scalers_path="./scalers"
):
    df_proc = df.copy()
    df_proc["physics_base"] = df_proc.apply(base_physics_model, axis=1)
    df_proc["residual"] = df_proc["y_strength_mpa"] - df_proc["physics_base"]

    X = df_proc[FEATURE_COLS].values
    y = df_proc["residual"].values

    idx_train, idx_val = train_test_split(
        np.arange(len(df_proc)), test_size=val_size, random_state=seed, shuffle=True
    )

    scaler_x = StandardScaler().fit(X[idx_train])
    scaler_y = StandardScaler().fit(y[idx_train].reshape(-1, 1))

    p_scalers = Path(scalers_path)
    p_scalers.mkdir(exist_ok=True)
    joblib.dump(scaler_x, p_scalers / "scaler_x.joblib")
    joblib.dump(scaler_y, p_scalers / "scaler_y.joblib")

    Xs = torch.tensor(scaler_x.transform(X), dtype=torch.float32)
    ys = torch.tensor(scaler_y.transform(y.reshape(-1, 1)).ravel(), dtype=torch.float32)

    ds_train = TensorDataset(Xs[idx_train], ys[idx_train])
    ds_val = TensorDataset(Xs[idx_val], ys[idx_val])

    # ExactGP prefers full-batch per split
    dl_train = DataLoader(ds_train, batch_size=len(ds_train), shuffle=False)
    dl_val = DataLoader(ds_val, batch_size=len(ds_val), shuffle=False)

    # Keep the split for later inference with original scaling
    return dl_train, dl_val, df_proc, scaler_x, scaler_y, idx_train, idx_val


def load_scalers(scalers_path="./scalers"):
    p_scalers = Path(scalers_path)
    scaler_x = joblib.load(p_scalers / "scaler_x.joblib")
    scaler_y = joblib.load(p_scalers / "scaler_y.joblib")
    return scaler_x, scaler_y


def load_df(data_dir="data"):
    p = Path(data_dir)
    concrete = json.loads(Path(p, "concrete.json").read_text())
    mortars = json.loads(Path(p, "mortar.json").read_text())
    rocks = json.loads(Path(p, "rock.json").read_text())

    rows = []
    for rec in concrete.values():
        m = mortars[rec["mortar_id"]]["properties"]
        r = rocks[rec["rock_id"]]["properties"]
        rows.append(
            {
                "mortar_splitting_strength_mpa": m["splitting_strength_mpa"],
                "mortar_shrinkage_in": m["shrinkage_inches"],
                "mortar_flexural_strength_mpa": m["flexural_strength_mpa"],
                "mortar_slump_in": m["slump_inches"],
                "mortar_compressive_strength_mpa": m["compressive_strength_mpa"],
                "mortar_poissons_ratio": m["poissons_ratio"],
                "rock_compressive_strength_mpa": r["compressive_strength_mpa"],
                "rock_size_in": r["size_inches"],
                "rock_density_lb_ft3": r["density_lb_ft3"],
                "rock_specific_gravity": r["specific_gravity"],
                "rock_ratio": rec["rock_ratio"],
                "y_strength_mpa": rec["concrete_compressive_strength_mpa"],
            }
        )
    return pd.DataFrame(rows)


# def make_dataloader(df: pd.DataFrame):
#     df = df.copy()
#     df["hirsch"] = df.apply(hirsch_row, axis=1)
#     df["residual"] = df["y_strength_mpa"] - df["hirsch"]

#     X = df[FEATURE_COLS].values
#     y = df["residual"].values

#     scaler_x = StandardScaler().fit(X)
#     scaler_y = StandardScaler().fit(y.reshape(-1, 1))

#     Xs = torch.tensor(scaler_x.transform(X), dtype=torch.float32)
#     ys = torch.tensor(scaler_y.transform(y.reshape(-1, 1)).ravel(), dtype=torch.float32)

#     ds = TensorDataset(Xs, ys)
#     # full-batch for ExactGP
#     dl = DataLoader(ds, batch_size=len(ds), shuffle=False)
#     return dl, df, scaler_x, scaler_y
