import lightning as L
import numpy as np
import torch


class PhysicsNN(L.LightningModule):
    def __init__(self, input_dim, scaler_x, scaler_y, hidden_dim=64, n_hidden_layers=4):
        super().__init__()
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.save_hyperparameters(ignore=["scaler_x", "scaler_y"])

        layers = [torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU()]
        for _ in range(n_hidden_layers):
            layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()])
        layers.append(torch.nn.Linear(hidden_dim, 1))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(-1)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(-1)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        rmse = torch.sqrt(loss)
        self.log("val_rmse", rmse, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def predict_residual(self, x_scaled):
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self(x_scaled.to(self.device))

        mean_res = pred_scaled.cpu().numpy()

        mean_res = self.scaler_y.inverse_transform(mean_res.reshape(-1, 1)).ravel()

        std_res = np.zeros_like(mean_res)
        return mean_res, std_res
