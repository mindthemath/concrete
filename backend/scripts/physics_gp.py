import gpytorch
import lightning as L
import torch


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
