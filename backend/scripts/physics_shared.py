from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

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


def base_physics_model(row):
    """Calculate the physics-based baseline prediction."""
    # Currently using Hirsch model, but this can be swapped for any physics model
    return hirsch_row(row)


def hirsch_row(row, eta=0.1):
    fm = row["mortar_compressive_strength_mpa"]
    fr = row["rock_compressive_strength_mpa"]
    r = row["rock_ratio"]
    return (1 - r) * fm + r * fr + eta * r * (1 - r) * (fr - fm)


# interface for the model type - something that implements the predict_residual method
# and must contain a scaler_x / scaler_y attribute which implement transform
class PhysicsModel(ABC):
    @abstractmethod
    def predict_residual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def scaler_x(self) -> StandardScaler:
        pass

    @abstractmethod
    def scaler_y(self) -> StandardScaler:
        pass


def combine_physics_with_residual(df_infer: pd.DataFrame, model: PhysicsModel):
    df = df_infer.copy()
    df["physics_base"] = df.apply(base_physics_model, axis=1)
    X = df[FEATURE_COLS].values
    Xs = torch.tensor(
        model.scaler_x.transform(X), dtype=torch.float32, device=model.device
    )
    mean_res, std_res = model.predict_residual(Xs)
    df["pred_mpa"] = df["physics_base"].values + mean_res
    df["pred_lo_mpa"] = df["pred_mpa"] - std_res
    df["pred_hi_mpa"] = df["pred_mpa"] + std_res
    return df[["physics_base", "pred_mpa", "pred_lo_mpa", "pred_hi_mpa", "residual"]]
