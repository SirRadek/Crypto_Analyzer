"""Reusable LSTM components adapted from the BTCLSTM project.

The original repository (https://github.com/SirRadek/BTCLSTM) provides a simple
PyTorch pipeline for modelling BTC price movements.  This module ports the core
building blocks into :mod:`crypto_analyzer` while keeping the codebase
self-contained and test friendly.  Three abstractions are exposed:

``SequenceDataset``
    Produces sliding window sequences from tabular data so temporal models can
    consume chronologically ordered observations.
``ModelLSTM``
    A compact recurrent neural network implemented as a :class:`torch.nn.Module`.
``LSTMTrainer``
    Orchestrates mini-batch training, validation splitting and a couple of
    lightweight diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

__all__ = [
    "SequenceConfig",
    "SequenceDataset",
    "ModelConfig",
    "ModelLSTM",
    "TrainingConfig",
    "LSTMTrainer",
]


@dataclass(slots=True)
class SequenceConfig:
    """Configuration for generating fixed-length sequences."""

    sequence_length: int = 48
    """Number of timesteps fed into the LSTM at once."""

    prediction_horizon: int = 1
    """Steps ahead the target value should be sampled from."""

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")


class SequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Create sliding-window sequences from tabular data."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        config: SequenceConfig,
    ) -> None:
        if len(features) != len(targets):
            raise ValueError("features and targets must have the same number of rows")
        self.features = np.asarray(features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.config = config

    def __len__(self) -> int:
        length = (
            len(self.features)
            - self.config.sequence_length
            - self.config.prediction_horizon
            + 1
        )
        return max(length, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = idx + self.config.sequence_length
        target_idx = end + self.config.prediction_horizon - 1
        if target_idx >= len(self.targets):
            raise IndexError("Index out of bounds for target sequence")
        x = self.features[start:end]
        y = self.targets[target_idx]
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.float32)


@dataclass(slots=True)
class ModelConfig:
    """Hyper-parameters describing the LSTM architecture."""

    input_size: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1)")


class ModelLSTM(nn.Module):
    """Simple LSTM-based regressor for crypto time series."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, max(config.hidden_size // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(config.hidden_size // 2, 1), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        preds = self.head(last_hidden)
        return preds.squeeze(-1)


@dataclass(slots=True)
class TrainingConfig:
    """Bundle configuration for :class:`LSTMTrainer`."""

    sequence: SequenceConfig
    model: ModelConfig
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    train_split: float = 0.8

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 < self.train_split < 1.0:
            raise ValueError("train_split must be between 0 and 1")
        if self.lr <= 0:
            raise ValueError("lr must be positive")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_dir = np.sign(np.diff(y_true, prepend=y_true[0]))
    pred_dir = np.sign(np.diff(y_pred, prepend=y_pred[0]))
    return float((true_dir == pred_dir).mean())


class LSTMTrainer:
    """High level helper orchestrating model training."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModelLSTM(config.model).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def _prepare_loaders(
        self, features: np.ndarray, targets: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        dataset = SequenceDataset(features, targets, self.config.sequence)
        if len(dataset) < 2:
            raise ValueError(
                "Not enough data to create sequences. Increase the input length or "
                "decrease sequence_length/prediction_horizon.")
        train_len = max(int(len(dataset) * self.config.train_split), 1)
        val_len = len(dataset) - train_len
        if val_len == 0:
            val_len = 1
            train_len = max(train_len - 1, 1)
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        return train_loader, val_loader

    def fit(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        train_loader, val_loader = self._prepare_loaders(features, targets)
        history: Dict[str, float | list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        for _ in range(self.config.epochs):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            train_loss /= max(len(train_loader.dataset), 1)

            self.model.eval()
            val_loss = 0.0
            preds_list: list[np.ndarray] = []
            targets_list: list[np.ndarray] = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    preds = self.model(batch_x)
                    loss = self.criterion(preds, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
                    preds_list.append(preds.cpu().numpy())
                    targets_list.append(batch_y.cpu().numpy())
            val_loss /= max(len(val_loader.dataset), 1)

            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))

        if preds_list and targets_list:
            predictions = np.concatenate(preds_list)
            true_values = np.concatenate(targets_list)
            history.update(
                {
                    "rmse": _rmse(true_values, predictions),
                    "mape": _mape(true_values, predictions),
                    "directional_accuracy": _directional_accuracy(true_values, predictions),
                }
            )
        return {k: float(v[-1]) if isinstance(v, list) else float(v) for k, v in history.items()}  # type: ignore[arg-type]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return predictions for each available sequence in ``features``."""

        dummy_targets = np.zeros(len(features), dtype=np.float32)
        dataset = SequenceDataset(features, dummy_targets, self.config.sequence)
        if len(dataset) == 0:
            return np.empty(0, dtype=np.float32)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        preds: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                preds.append(outputs.cpu().numpy())
        return np.concatenate(preds)
