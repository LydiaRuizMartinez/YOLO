import pytest
import torch
from torch.utils.data import Subset  # Import Subset
from src.data_manager import DataManager
from src.model_handler import ModelHandler
from src.utils import get_final_layer  # Import from utils now
from src.strategies import (
    LastLayerStrategy,
    FeatureExtractionStrategy,
    FullFineTuningStrategy,
)
from pathlib import Path


TEST_DATA_DIR = Path("./test_data_cache")


@pytest.fixture
def setup_environment():
    """
    A comprehensive fixture to set up a full environment for testing strategies.
    It return the model directly.
    """
    config = {
        "data": {"name": "CIFAR10", "path": str(TEST_DATA_DIR), "batch_size": 4},
        "model": {"name": "alexnet"},
        "training": {
            "epochs": 1,  # Only one epoch for fast testing
            "learning_rate": 1e-3,
        },
    }

    # 1. Setup DataManager
    dm = DataManager(config)
    dm.load_data()

    # Reduce the set's sizes for the tests
    train_subset_indices = range(200)  # Use 200 samples for training
    val_subset_indices = range(100)  # Use 100 samples for validation

    dm.train_dataset = Subset(dm.train_dataset, train_subset_indices)
    dm.val_dataset = Subset(dm.val_dataset, val_subset_indices)

    # 2. Setup ModelHandler
    mh = ModelHandler(config)
    mh.load_model()

    # Return the model object directly
    return dm, mh, config


def test_last_layer_strategy_param_freezing(setup_environment):
    """
    Tests if the LastLayerStrategy correctly freezes and unfreezes parameters.
    """
    _, model_handler, config = setup_environment  # Unpack the model directly
    strategy = LastLayerStrategy()

    model = model_handler.model
    # Pass the model directly to the prepare method
    prepared_model, _ = strategy._prepare_model_and_optimizer(model, config)

    # Find the final layer
    final_layer_name, final_layer = get_final_layer(prepared_model)

    # Check that all parameters in the final layer are trainable
    for param in final_layer.parameters():
        assert param.requires_grad == True

    # Check that some parameters in the rest of the model are frozen
    first_param = next(prepared_model.parameters())
    assert first_param.requires_grad == False


def test_last_layer_strategy_runs(setup_environment):
    """
    Tests if the LastLayerStrategy runs through a full execute cycle without errors.
    """
    data_manager, model_handler, config = setup_environment  # Unpack the model directly
    strategy = LastLayerStrategy()

    train_loader = data_manager.get_train_loader()
    val_loader = data_manager.get_val_loader()

    # Modify model's last layer
    model_handler.replace_last_layer(data_manager.num_classes)

    # The test passes if this runs without crashing and returns a dictionary
    # Pass the model object directly to the execute method
    results = strategy.execute(model_handler.model, train_loader, val_loader, config)

    assert isinstance(results, dict)
    assert "accuracy" in results
    accuracy = results["accuracy"]
    print(f"Accuracy with a sample subset: {accuracy:.2f} ")
    assert "loss" in results


def test_feature_extraction_strategy_runs(setup_environment):
    """
    Tests if the FeatureExtractionStrategy runs its full cycle without errors.
    It uses the unprepared model.
    """

    data_manager, model_handler, config = setup_environment
    strategy = FeatureExtractionStrategy()

    train_loader = data_manager.get_train_loader()
    val_loader = data_manager.get_val_loader()

    model = model_handler.model
    # The test passes if this runs without crashing and returns a dictionary
    results = strategy.execute(model, train_loader, val_loader, config)

    assert isinstance(results, dict)
    assert "accuracy" in results
    # This strategy doesn't produce a 'loss' key, so we don't check for it.


def test_full_finetuning_strategy_unfreezes_all_parameters(setup_environment):
    """
    Tests if the FullFineTuningStrategy correctly unfreezes all parameters in the model.

    This test verifies that:
    1. All parameters in the model have requires_grad=True after preparation
    2. The optimizer contains all model parameters
    3. No parameters are frozen anywhere in the model
    """
    _, model_handler, config = setup_environment
    strategy = FullFineTuningStrategy()

    model = model_handler.model

    # Count initial parameters (for reference)
    total_params = sum(p.numel() for p in model.parameters())

    # Prepare the model with the strategy
    prepared_model, optimizer = strategy._prepare_model_and_optimizer(model, config)

    # Test 1: Check that ALL parameters are trainable
    trainable_params = []
    frozen_params = []

    for name, param in prepared_model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    # Assert no parameters are frozen
    assert len(frozen_params) == 0, f"Found frozen parameters: {frozen_params}"

    # Assert all parameters are trainable
    assert len(trainable_params) > 0, "No trainable parameters found"

    # Test 2: Count trainable parameters matches total parameters
    trainable_param_count = sum(
        p.numel() for p in prepared_model.parameters() if p.requires_grad
    )
    assert (
        trainable_param_count == total_params
    ), f"Mismatch: {trainable_param_count} trainable vs {total_params} total parameters"

    # Test 3: Verify optimizer contains all parameters
    optimizer_param_count = sum(
        p.numel() for group in optimizer.param_groups for p in group["params"]
    )
    assert (
        optimizer_param_count == total_params
    ), f"Optimizer has {optimizer_param_count} parameters, model has {total_params}"

    # Test 4: Verify specific layers are trainable (spot check)
    # Check first layer (usually features or conv layers)
    first_param = next(prepared_model.parameters())
    assert (
        first_param.requires_grad == True
    ), "First layer parameter should be trainable"

    # Check last layer
    *_, last_param = prepared_model.parameters()
    assert last_param.requires_grad == True, "Last layer parameter should be trainable"


def test_full_finetuning_strategy_execution(setup_environment):
    """
    Tests if the FullFineTuningStrategy runs through
    a full training cycle without errors.

    This test verifies that:
    1. The strategy can execute a complete training epoch
    2. It returns proper metrics (accuracy and loss)
    3. The model is actually being updated during training
    """
    data_manager, model_handler, config = setup_environment

    # modify learning rate because weights are already trained
    config["training"]["learning_rate"] = 1e-4
    strategy = FullFineTuningStrategy()

    train_loader = data_manager.get_train_loader()
    val_loader = data_manager.get_val_loader()

    # Modify model's last layer to match dataset classes
    model_handler.replace_last_layer(data_manager.num_classes)

    model = model_handler.model

    # Store initial state of some parameters to verify they change
    initial_first_param = next(model.parameters()).clone().detach()

    # Execute the strategy
    results = strategy.execute(model, train_loader, val_loader, config)

    # Test 1: Verify results structure
    assert isinstance(results, dict), "Results should be a dictionary"
    assert "accuracy" in results, "Results should contain 'accuracy'"
    assert "loss" in results, "Results should contain 'loss'"

    # Test 2: Verify metrics are reasonable
    accuracy = results["accuracy"]
    loss = results["loss"]

    assert 0 <= accuracy <= 100, f"Accuracy {accuracy} is out of valid range [0, 100]"
    assert loss >= 0, f"Loss {loss} should be non-negative"

    print(f"Full fine-tuning - Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")

    # Test 3: Verify parameters have actually changed (model was trained)
    final_first_param = next(model.parameters()).clone().detach()
    param_changed = not torch.allclose(initial_first_param, final_first_param)
    assert param_changed, "Model parameters should have changed after training"


def test_full_finetuning_vs_last_layer_parameter_count(setup_environment):
    """
    Comparative test to verify that FullFineTuningStrategy trains more parameters
    than LastLayerStrategy.

    This test ensures that the full fine-tuning strategy indeed trains all parameters,
    while last layer strategy only trains a subset.
    """
    from src.strategies import LastLayerStrategy

    _, model_handler, config = setup_environment

    # Setup for FullFineTuningStrategy
    full_strategy = FullFineTuningStrategy()
    model_full = model_handler.model

    # Reload model for LastLayerStrategy (to have a fresh copy)
    model_handler.load_model()
    last_strategy = LastLayerStrategy()
    model_last = model_handler.model

    # Prepare both models
    model_full_prepared, _ = full_strategy._prepare_model_and_optimizer(
        model_full, config
    )
    model_last_prepared, _ = last_strategy._prepare_model_and_optimizer(
        model_last, config
    )

    # Count trainable parameters in each
    full_trainable = sum(
        p.numel() for p in model_full_prepared.parameters() if p.requires_grad
    )
    last_trainable = sum(
        p.numel() for p in model_last_prepared.parameters() if p.requires_grad
    )

    # Assert full fine-tuning has more trainable parameters
    assert full_trainable > last_trainable, (
        f"Full fine-tuning ({full_trainable}) should have more trainable parameters "
        f"than last layer only ({last_trainable})"
    )

    print(f"Full fine-tuning: {full_trainable:,} trainable parameters")
    print(f"Last layer only: {last_trainable:,} trainable parameters")
    print(f"Difference: {full_trainable - last_trainable:,} additional parameters")
