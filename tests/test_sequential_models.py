import numpy as np
import torch

from crypto_analyzer.models import sequential


def test_sequence_dataset_length_and_item():
    config = sequential.WindowConfig(window=4, horizon=1)
    features = np.arange(30).reshape(10, 3)
    targets = np.linspace(0, 1, 10)
    dataset = sequential.SequenceWindowDataset(features, targets, config)
    assert len(dataset) == 6
    x, y = dataset[0]
    assert x.shape == (4, 3)
    assert isinstance(y.item(), float)


def test_train_tcn_classifier_cpu():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(64, 5)).astype(np.float32)
    targets = (rng.random(64) > 0.6).astype(np.float32)
    training = sequential.TrainingConfig(
        window=sequential.WindowConfig(window=8, horizon=1),
        epochs=1,
        batch_size=16,
        lr=1e-3,
        device="cpu",
    )
    model, metrics = sequential.train_sequence_classifier(
        features,
        targets,
        training=training,
        model="tcn",
        tcn_config=sequential.TCNConfig(input_channels=5, hidden_channels=8, num_blocks=2),
    )
    assert isinstance(model, torch.nn.Module)
    assert any(key.endswith("train_loss") for key in metrics)


def test_train_transformer_classifier_cpu():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(80, 4)).astype(np.float32)
    targets = (rng.random(80) > 0.4).astype(np.float32)
    training = sequential.TrainingConfig(
        window=sequential.WindowConfig(window=6, horizon=1),
        epochs=1,
        batch_size=8,
        lr=1e-3,
        device="cpu",
    )
    model, metrics = sequential.train_sequence_classifier(
        features,
        targets,
        training=training,
        model="transformer",
        transformer_config=sequential.TransformerConfig(input_dim=4, hidden_dim=16, num_layers=1, num_heads=2),
    )
    assert isinstance(model, torch.nn.Module)
    assert any(key.endswith("val_accuracy") for key in metrics)

