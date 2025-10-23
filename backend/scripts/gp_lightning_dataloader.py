# gp_lightning_dataloader.py
import json
from pathlib import Path

import gpytorch
import joblib
import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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


def make_dataloaders_with_split(
    df: pd.DataFrame, val_size=0.2, seed=42, scalers_path="./scalers"
):
    df_proc = df.copy()
    df_proc["hirsch"] = df_proc.apply(hirsch_row, axis=1)
    df_proc["residual"] = df_proc["y_strength_mpa"] - df_proc["hirsch"]

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


def hirsch_row(row, eta=0.1):
    fm = row["mortar_compressive_strength_mpa"]
    fr = row["rock_compressive_strength_mpa"]
    r = row["rock_ratio"]
    return (1 - r) * fm + r * fr + eta * r * (1 - r) * (fr - fm)


class ExactResidualGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class PhysicsGPR(L.LightningModule):
    def __init__(self, scaler_x, scaler_y, train_x=None, train_y=None):
        super().__init__()
        self.save_hyperparameters(ignore=["scaler_x", "scaler_y"])
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp = None
        self.mll = None
        self.train_x = train_x
        self.train_y = train_y

    def load_state_dict(self, state_dict, strict: bool = True):
        # Handle GP and MLL state loading manually
        gp_state = {}
        mll_state = {}
        other_state = {}

        for key, value in state_dict.items():
            if key.startswith("gp."):
                gp_state[key[3:]] = value  # Remove 'gp.' prefix
            elif key.startswith("mll."):
                mll_state[key[4:]] = value  # Remove 'mll.' prefix
            else:
                other_state[key] = value

        # Load other parameters first using super()
        super().load_state_dict(other_state, strict=False)

        # Initialize GP and MLL if we have training data
        if self.train_x is not None and self.train_y is not None:
            self.setup_gp(self.train_x, self.train_y)

            # Load GP state if available
            if gp_state and self.gp is not None:
                self.gp.load_state_dict(gp_state)

            # Load MLL state if available
            if mll_state and self.mll is not None:
                self.mll.load_state_dict(mll_state)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)

    def setup_gp(self, x, y):
        """Initialize GP and MLL with training data"""
        if self.gp is None:
            self.train_x = x.clone()
            self.train_y = y.clone()
            self.gp = ExactResidualGP(self.train_x, self.train_y, self.likelihood)
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood, self.gp
            )
            self.gp = self.gp.to(self.device)
            self.likelihood = self.likelihood.to(self.device)

    def training_step(self, batch, batch_idx):
        x, y = batch  # full batch
        # Initialize GP on first step (needs train_x reference inside ExactGP)
        if self.gp is None:
            self.setup_gp(x, y)

        self.gp.train()
        self.likelihood.train()

        # The GP is already initialized with training data, so we can call it
        output = self.gp(x)
        loss = -self.mll(output, y)
        self.log("train_nll", loss, prog_bar=True, on_epoch=True)
        return loss

    def predict_residual(self, x_scaled):
        self.gp.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.gp(x_scaled))
        mean_res = pred.mean.cpu().numpy()
        std_res = pred.stddev.cpu().numpy()
        # inverse-scale residual
        mean_res = self.scaler_y.inverse_transform(mean_res.reshape(-1, 1)).ravel()
        std_res = self.scaler_y.inverse_transform(std_res.reshape(-1, 1)).ravel()
        return mean_res, std_res

    def on_save_checkpoint(self, ckpt):
        # Save GP and MLL states separately with prefixes
        if self.gp is not None:
            gp_state = self.gp.state_dict()
            for key, value in gp_state.items():
                ckpt[f"gp.{key}"] = value

        if self.mll is not None:
            mll_state = self.mll.state_dict()
            for key, value in mll_state.items():
                ckpt[f"mll.{key}"] = value

        # Save other parameters
        state_dict = self.state_dict()
        for key, value in state_dict.items():
            if not key.startswith(("gp.", "mll.")):
                ckpt[key] = value

        ckpt["scaler_x"] = self.scaler_x
        ckpt["scaler_y"] = self.scaler_y
        ckpt["train_x"] = self.train_x
        ckpt["train_y"] = self.train_y

    def on_load_checkpoint(self, ckpt):
        self.train_x = ckpt.get("train_x")
        self.train_y = ckpt.get("train_y")
        # Load the scalers
        if "scaler_x" in ckpt:
            self.scaler_x = ckpt["scaler_x"]
        if "scaler_y" in ckpt:
            self.scaler_y = ckpt["scaler_y"]

    def on_train_start(self):
        """Initialize GP if we have training data from checkpoint"""
        if self.train_x is not None and self.train_y is not None and self.gp is None:
            self.setup_gp(self.train_x, self.train_y)

    def validation_step(self, batch, batch_idx):
        # If GP not initialized yet, initialize with training data if available
        if self.gp is None and self.train_x is not None and self.train_y is not None:
            self.setup_gp(self.train_x, self.train_y)
        elif self.gp is None:
            return {
                "val_rmse": torch.tensor(float("inf")),
                "val_nll": torch.tensor(float("inf")),
            }

        self.gp.eval()
        self.likelihood.eval()
        x, y = batch
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.gp(x))
            mean = pred.mean
            rmse = torch.sqrt(torch.mean((mean - y) ** 2))
            # Negative predictive log likelihood on val (per point)
            nll = -pred.log_prob(y) / y.numel()

        # Log both; checkpoint can monitor val_rmse
        self.log("val_rmse", rmse, prog_bar=True, on_epoch=True)
        self.log("val_nll", nll, prog_bar=False, on_epoch=True)
        return {"val_rmse": rmse, "val_nll": nll}


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


def combine_hirsch_with_residual(df_infer: pd.DataFrame, model: PhysicsGPR):
    df = df_infer.copy()
    df["hirsch"] = df.apply(hirsch_row, axis=1)
    X = df[FEATURE_COLS].values
    Xs = torch.tensor(
        model.scaler_x.transform(X), dtype=torch.float32, device=model.device
    )
    mean_res, std_res = model.predict_residual(Xs)
    df["pred_mpa"] = df["hirsch"].values + mean_res
    df["pred_lo_mpa"] = df["pred_mpa"] - std_res
    df["pred_hi_mpa"] = df["pred_mpa"] + std_res
    return df[["hirsch", "pred_mpa", "pred_lo_mpa", "pred_hi_mpa", "residual"]]
