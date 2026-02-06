import pytest
import torch
from src.data_manager import DataManager
from src.model_handler import ModelHandler
from src.utils import get_final_layer  # Import from utils now
from src.strategies import (
    LastLayerStrategy,
    NoTrainingStrategy,
    FeatureExtractionStrategy,
    FullFineTuningStrategy,
)
from pathlib import Path


TEST_DATA_DIR = Path("./test_data_cache")


def test_resnet_cifar10_last_layer_fine_tuning():
    config = {
        "data": {"name": "CIFAR10", "path": str(TEST_DATA_DIR), "batch_size": 4},
        "model": {"name": "resnet18"},
        "training": {
            "epochs": 5,
            "learning_rate": 1e-3,
        },
    }

    # 1. Load data
    data_manager = DataManager(config)
    data_manager.load_data()
    train_loader = data_manager.get_train_loader()
    val_loader = data_manager.get_val_loader()

    # 2. Load model and replace last layer
    model_handler = ModelHandler(config)
    model_handler.load_model()
    model_handler.replace_last_layer(data_manager.num_classes)

    # 3. Train and evaluate
    strategy = LastLayerStrategy()
    results = strategy.execute(model_handler.model, train_loader, val_loader, config)

    assert isinstance(results, dict)
    assert "accuracy" in results
    accuracy = results["accuracy"]
    print(f"Accuracy: {accuracy:.2f} ")
    assert "loss" in results


def test_resnet_vgg16_full_fine_tuning():
    config = {
        "data": {"name": "CIFAR10", "path": str(TEST_DATA_DIR), "batch_size": 4},
        "model": {"name": "resnet18"},
        "training": {
            "epochs": 5,
            "learning_rate": 1e-4,
        },
    }

    # 1. Load data
    data_manager = DataManager(config)
    data_manager.load_data()
    train_loader = data_manager.get_train_loader()
    val_loader = data_manager.get_val_loader()

    # 2. Load model and replace last layer
    model_handler = ModelHandler(config)
    model_handler.load_model()
    model_handler.replace_last_layer(data_manager.num_classes)

    # 3. Train and evaluate
    strategy = FullFineTuningStrategy()
    results = strategy.execute(model_handler.model, train_loader, val_loader, config)

    assert isinstance(results, dict)
    assert "accuracy" in results
    accuracy = results["accuracy"]
    print(f"Accuracy: {accuracy:.2f} ")
    assert "loss" in results


def test_resnet_vgg16_full_fine_tuning():
    config = {
        "data": {"name": "CIFAR10", "path": str(TEST_DATA_DIR), "batch_size": 4},
        "model": {"name": "resnet18"},
        "training": {
            "epochs": 5,
            "learning_rate": 1e-4,
        },
    }

    # 1. Load data
    data_manager = DataManager(config)
    data_manager.load_data()
    train_loader = data_manager.get_train_loader()
    val_loader = data_manager.get_val_loader()

    # 2. Load model and replace last layer
    model_handler = ModelHandler(config)
    model_handler.load_model()
    model_handler.replace_last_layer(data_manager.num_classes)

    # 3. Train and evaluate
    strategy = FullFineTuningStrategy()
    results = strategy.execute(model_handler.model, train_loader, val_loader, config)

    assert isinstance(results, dict)
    assert "accuracy" in results
    accuracy = results["accuracy"]
    print(f"Accuracy: {accuracy:.2f} ")
    assert "loss" in results
