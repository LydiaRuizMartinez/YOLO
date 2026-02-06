# tests/test_model_handler.py

import pytest
import torch.nn as nn
from src.model_handler import ModelHandler


@pytest.fixture
def model_handler_instance() -> ModelHandler:
    """
    Fixture to create a ModelHandler instance for testing.
    We'll use 'alexnet' as our test case.
    """
    test_config = {"model": {"name": "alexnet"}}
    return ModelHandler(config=test_config)


def test_model_loading(model_handler_instance):
    """
    Tests if alexnet model is loaded correctly.
    """
    # Load the model
    model_handler_instance.load_model()

    # Test 1: Model should not be None after loading
    assert (
        model_handler_instance.model is not None
    ), "Model should be loaded and not None"

    # Test 2: Check if it's the correct model type (AlexNet)
    assert (
        "alexnet" in str(type(model_handler_instance.model)).lower()
    ), f"Expected AlexNet model, got {type(model_handler_instance.model)}"


def test_replace_last_layer_alexnet(model_handler_instance):
    """
    Tests if the final layer is correctly replaced in AlexNet.
    """
    # Load the model first and store the original last layer
    model_handler_instance.load_model()
    original_last_layer = model_handler_instance.model.classifier[6]
    original_in_features = original_last_layer.in_features

    # Prepare it for a dataset with 10 classes (like MNIST or CIFAR10)
    num_test_classes = 10
    model_handler_instance.replace_last_layer(num_classes=num_test_classes)

    # Get the new final layer
    new_layer = model_handler_instance.model.classifier[6]

    # Test 1: The layer should be a new object (different memory address)
    assert id(original_last_layer) != id(
        new_layer
    ), "The final layer should be replaced with a new instance"

    # Test 2: The new layer should be a Linear layer
    assert isinstance(new_layer, nn.Linear), "The new final layer should be nn.Linear"

    # Test 3: The number of output features should match the specified classes
    assert (
        new_layer.out_features == num_test_classes
    ), f"Expected {num_test_classes} output features, got {new_layer.out_features}"

    # Test 4: The number of input features should be preserved from original
    assert (
        new_layer.in_features == original_in_features
    ), f"Input features should be preserved: expected {original_in_features}, \
              got {new_layer.in_features}"

    # Test 5: For AlexNet, input features should be 4096
    assert (
        new_layer.in_features == 4096
    ), f"AlexNet final layer should have 4096 input features, \
              got {new_layer.in_features}"


def test_replace_last_layer_inception():
    """
    Tests Inception models loads correctly
    """

    num_test_classes = 10

    test_config = {"model": {"name": "inception"}}
    model_handler_instance = ModelHandler(config=test_config)
    model_handler_instance.load_model()

    # Verify model loaded
    assert model_handler_instance.model is not None, "Inception model should be loaded"
    assert (
        "inception" in str(type(model_handler_instance.model)).lower()
    ), f"Expected Inception model, got {type(model_handler_instance.model)}"

    # Replace the last layer for the test classes
    model_handler_instance.replace_last_layer(num_test_classes)

    # Test forward pass with inception's expected input size
    import torch

    test_input = torch.randn(2, 3, 299, 299)  # Inception expects 299x299
    model_handler_instance.model.eval()  # Set to eval to avoid auxiliary outputs

    with torch.no_grad():
        output = model_handler_instance.model(test_input)

    # Check output shape
    assert output.shape == (
        2,
        num_test_classes,
    ), f"Expected output shape (2, {num_test_classes}), got {output.shape}"

    print(f"Inception model loaded successfully with {num_test_classes} output classes")


def test_replace_last_layer_resnet():
    """
    Tests if the final layer is correctly replaced in Resnet18
    """

    num_test_classes = 10

    test_config = {"model": {"name": "resnet18"}}
    model_handler_instance = ModelHandler(config=test_config)
    model_handler_instance.load_model()

    # Store the original final layer before replacement
    original_fc = model_handler_instance.model.fc
    original_in_features = original_fc.in_features

    # Replace the last layer
    model_handler_instance.replace_last_layer(num_classes=num_test_classes)

    # Get the new final layer
    new_fc = model_handler_instance.model.fc

    # Test 1: The layer should be a new object (different memory address)
    assert id(original_fc) != id(
        new_fc
    ), "The final layer should be replaced with a new instance"

    # Test 2: The new layer should be a Linear layer
    assert isinstance(new_fc, nn.Linear), "The new final layer should be nn.Linear"

    # Test 3: The number of output features should match the specified classes
    assert (
        new_fc.out_features == num_test_classes
    ), f"Expected {num_test_classes} output features, got {new_fc.out_features}"

    # Test 4: The number of input features should be preserved from original
    assert (
        new_fc.in_features == original_in_features
    ), f"Input features should be preserved: expected {original_in_features}, \
            got {new_fc.in_features}"

    # Test 5: For ResNet18, input features should be 512
    assert (
        new_fc.in_features == 512
    ), f"ResNet18 final layer should have 512 input features, got {new_fc.in_features}"


def test_unsupported_model():
    """
    Tests that the handler raises an error for a model not in the registry.
    The error should be raised during load_model(), not __init__().
    """
    # Create config with unsupported model name
    unsupported_config = {"model": {"name": "this_model_does_not_exist"}}
    handler = ModelHandler(config=unsupported_config)

    # Test: Use pytest.raises to check that a ValueError is correctly thrown
    with pytest.raises(ValueError, match="not found in internal MODEL_REGISTRY"):
        handler.load_model()
