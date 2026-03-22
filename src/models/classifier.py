"""XGBoost + PyTorch MLP ensemble classifier for wallet behavior classification.

Combines gradient-boosted trees (XGBoost) with a neural network (MLP) via a
weighted probability ensemble.  Hyperparameters for XGBoost are tuned with
Optuna (100 trials, TPE sampler).  Ensemble weights are optimized on a held-out
validation set to maximize macro F1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import optuna
import structlog
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from src.config import settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# PyTorch MLP
# ---------------------------------------------------------------------------


class WalletMLP(nn.Module):
    """Simple feed-forward classifier: input -> 128 -> 64 -> num_classes.

    Uses ReLU activations, BatchNorm, and Dropout(0.3) between hidden layers.
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(x)
        return result


# ---------------------------------------------------------------------------
# Ensemble classifier
# ---------------------------------------------------------------------------


class WalletClassifier:
    """Ensemble of XGBoost and MLP for wallet-type classification.

    Parameters
    ----------
    num_classes : int
        Number of target classes (default 7).
    device : str
        PyTorch device string (``'cpu'`` or ``'cuda'``).
    """

    _SEED: ClassVar[int] = settings.random_seed

    def __init__(self, num_classes: int = 7, device: str = "cpu") -> None:
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.xgb_model: XGBClassifier | None = None
        self.mlp_model: WalletMLP | None = None
        self.input_dim: int | None = None
        self.ensemble_weights: np.ndarray = np.array([0.5, 0.5])

    # ------------------------------------------------------------------
    # XGBoost with Optuna
    # ------------------------------------------------------------------

    def train_xgboost(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 100,
    ) -> XGBClassifier:
        """Train an XGBoost classifier with Optuna hyperparameter search.

        Uses the TPE sampler and optimizes macro F1 on the validation set.
        """
        logger.info("xgboost_tuning_start", n_trials=n_trials)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            model = XGBClassifier(
                **params,
                objective="multi:softprob",
                num_class=self.num_classes,
                tree_method="hist",
                random_state=self._SEED,
                eval_metric="mlogloss",
                early_stopping_rounds=20,
                verbosity=0,
            )
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                verbose=False,
            )
            preds = model.predict(x_val)
            return float(f1_score(y_val, preds, average="macro"))

        sampler = optuna.samplers.TPESampler(seed=self._SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        logger.info(
            "xgboost_tuning_done",
            best_f1=study.best_value,
            best_params=best,
        )

        # Re-train with best params on train data with early stopping on val
        self.xgb_model = XGBClassifier(
            **best,
            objective="multi:softprob",
            num_class=self.num_classes,
            tree_method="hist",
            random_state=self._SEED,
            eval_metric="mlogloss",
            early_stopping_rounds=20,
            verbosity=0,
        )
        self.xgb_model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        return self.xgb_model

    # ------------------------------------------------------------------
    # MLP training
    # ------------------------------------------------------------------

    def train_mlp(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> WalletMLP:
        """Train the MLP with AdamW + cosine-annealing LR schedule."""
        self.input_dim = x_train.shape[1]
        self.mlp_model = WalletMLP(self.input_dim, self.num_classes).to(self.device)

        train_ds = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.mlp_model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_f1 = 0.0
        best_state: dict | None = None

        logger.info("mlp_training_start", epochs=epochs, lr=lr, batch_size=batch_size)

        for epoch in range(1, epochs + 1):
            # --- train ---
            self.mlp_model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.mlp_model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            scheduler.step()

            # --- validate ---
            self.mlp_model.eval()
            all_preds: list[np.ndarray] = []
            with torch.no_grad():
                for xb, _yb in val_loader:
                    xb = xb.to(self.device)
                    logits = self.mlp_model(xb)
                    all_preds.append(logits.argmax(dim=1).cpu().numpy())

            val_preds = np.concatenate(all_preds)
            val_f1 = f1_score(y_val, val_preds, average="macro")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in self.mlp_model.state_dict().items()}

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "mlp_epoch",
                    epoch=epoch,
                    train_loss=total_loss / len(train_ds),
                    val_f1=val_f1,
                )

        # Restore best checkpoint
        if best_state is not None:
            self.mlp_model.load_state_dict(best_state)
        logger.info("mlp_training_done", best_val_f1=best_val_f1)
        return self.mlp_model

    # ------------------------------------------------------------------
    # Ensemble weight optimization
    # ------------------------------------------------------------------

    def optimize_ensemble_weights(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> np.ndarray:
        """Find the ensemble weight vector that maximizes macro F1 on *x_val*.

        Returns the optimized weight array ``[w_xgb, w_mlp]`` (sums to 1).
        """
        xgb_proba = self._xgb_predict_proba(x_val)
        mlp_proba = self._mlp_predict_proba(x_val)

        def neg_f1(w: np.ndarray) -> float:
            w_norm = w / w.sum()
            combined = w_norm[0] * xgb_proba + w_norm[1] * mlp_proba
            preds = combined.argmax(axis=1)
            return float(-f1_score(y_val, preds, average="macro"))

        result = minimize(
            neg_f1,
            x0=np.array([0.5, 0.5]),
            method="Nelder-Mead",
            options={"maxiter": 200},
        )
        self.ensemble_weights = result.x / result.x.sum()
        logger.info(
            "ensemble_weights_optimized",
            weights=self.ensemble_weights.tolist(),
            val_f1=-result.fun,
        )
        return self.ensemble_weights

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return ensemble class probabilities of shape ``(n, num_classes)``.

        Falls back to XGBoost-only if the MLP has not been trained.
        """
        xgb_proba = self._xgb_predict_proba(x)
        if self.mlp_model is None:
            return xgb_proba
        mlp_proba = self._mlp_predict_proba(x)
        w = self.ensemble_weights
        result: np.ndarray = w[0] * xgb_proba + w[1] * mlp_proba
        return result

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(predicted_labels, confidence_scores)``.

        Confidence score is the maximum ensemble probability for each sample.
        """
        proba = self.predict_proba(x)
        labels = proba.argmax(axis=1)
        confidence = proba.max(axis=1)
        return labels, confidence

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _xgb_predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.xgb_model is None:
            raise RuntimeError("XGBoost model has not been trained yet.")
        return self.xgb_model.predict_proba(x)

    def _mlp_predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.mlp_model is None:
            raise RuntimeError("MLP model has not been trained yet.")
        self.mlp_model.eval()
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            logits = self.mlp_model(tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the full ensemble to *path* (a directory)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # XGBoost
        if self.xgb_model is not None:
            self.xgb_model.save_model(str(path / "xgb_model.json"))

        # MLP
        if self.mlp_model is not None:
            torch.save(self.mlp_model.state_dict(), path / "mlp_model.pt")

        # Metadata
        meta = {
            "num_classes": self.num_classes,
            "input_dim": self.input_dim,
            "ensemble_weights": self.ensemble_weights.tolist(),
            "device": str(self.device),
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("model_saved", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> WalletClassifier:
        """Load a previously saved ensemble from *path*."""
        path = Path(path)

        with open(path / "meta.json") as f:
            meta = json.load(f)

        obj = cls(
            num_classes=meta["num_classes"],
            device=meta.get("device", "cpu"),
        )
        obj.input_dim = meta["input_dim"]
        obj.ensemble_weights = np.array(meta["ensemble_weights"])

        # XGBoost
        xgb_path = path / "xgb_model.json"
        if xgb_path.exists():
            obj.xgb_model = XGBClassifier()
            obj.xgb_model.load_model(str(xgb_path))

        # MLP
        mlp_path = path / "mlp_model.pt"
        if mlp_path.exists() and obj.input_dim is not None:
            obj.mlp_model = WalletMLP(obj.input_dim, obj.num_classes).to(obj.device)
            state = torch.load(mlp_path, map_location=obj.device, weights_only=True)
            obj.mlp_model.load_state_dict(state)
            obj.mlp_model.eval()

        logger.info("model_loaded", path=str(path))
        return obj
