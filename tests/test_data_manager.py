import pytest
import torch
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


from src.data_manager import DataManager

# Define a constant for the test data directory at the top of the file.
# This keeps the path in one easy-to-find place for this test module.
TEST_DATA_DIR = Path("./test_data_cache")


@pytest.fixture
def data_manager_instance() -> DataManager:
    """
    fixture tu use MNIST for testing
    """

    # Ensure the cache directory exists
    TEST_DATA_DIR.mkdir(exist_ok=True)

    # We use "CIFAR" for fast real testing
    test_config = {
        "data": {
            "name": "CIFAR10",
            "path": str(TEST_DATA_DIR),  # Usamos el directorio temporal
            "batch_size": 4,
        }
    }

    dm = DataManager(config=test_config)
    return dm


@pytest.mark.order(1)
def test_get_transforms(data_manager_instance):
    """Test that _get_transforms returns a valid transformation pipeline.

    Tests that the transformation pipeline:
    1. Returns a transforms.Compose object
    2. Contains the expected number of transformations
    3. Can successfully transform a sample tensor
    4. Produces output with correct shape and normalization range
    """

    # You call fixtures by name
    data_manager = data_manager_instance

    # Get the transform pipeline
    transform = data_manager._get_transforms()

    # Test 1: Check it is a transforms instance
    assert isinstance(transform, transforms.Compose)

    # Check it has the expected transformations
    # Expected: Resize, ToTensor, Normalize (3 transforms)
    assert len(transform.transforms) == 3

    # Check transformation types
    assert isinstance(
        transform.transforms[0], transforms.Resize
    ), "First transform should be resize"

    # Check the transformation pipeline works
    from PIL import Image
    import numpy as np

    # Create a dummy RGB image (224x224x3)
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )

    # Apply transforms
    transformed = transform(dummy_image)

    # Test 5: Check output properties
    assert isinstance(transformed, torch.Tensor), "Transform output should be a tensor"
    assert transformed.shape == (
        3,
        224,
        224,
    ), f"Expected shape (3, 224, 224), got {transformed.shape}"

    # Test 6: Check normalization (values should be roughly in range [-2, 2] after
    #  ImageNet normalization)
    assert transformed.min() >= -3.0, "Normalized values seem too low"
    assert transformed.max() <= 3.0, "Normalized values seem too high"


@pytest.mark.order(2)
def test_load_data(data_manager_instance):

    data_manager = data_manager_instance

    data_manager.load_data()

    # Check the number of classes is correct
    assert (
        data_manager.num_classes == 10
    ), f"CIFAR10 should have 10 classes, got {data_manager.num_classes}"

    # Check datasets have transforms applied
    assert (
        data_manager.train_dataset.transform is not None
    ), "Train dataset should have transforms"
    assert (
        data_manager.val_dataset.transform is not None
    ), "Val dataset should have transforms"

    # Check dataset types
    assert isinstance(
        data_manager.train_dataset, datasets.CIFAR10
    ), "Train dataset should be CIFAR10"
    assert isinstance(
        data_manager.val_dataset, datasets.CIFAR10
    ), "Val dataset should be CIFAR10"


@pytest.mark.order(3)
def test_loaders(data_manager_instance):

    data_manager = data_manager_instance

    # Load data
    data_manager.load_data()

    # Tests batches and image sizes are correct.
    train_loader = data_manager.get_train_loader()
    assert isinstance(
        train_loader.sampler, RandomSampler
    ), "Training Data should be shuffled"
    # torch.utils.data.sampler.RandomSampler

    val_loader = data_manager.get_val_loader()
    assert isinstance(
        val_loader.sampler, SequentialSampler
    ), "Validation Data should not be shuffled"

    loaders = [train_loader, val_loader]
    for loader in loaders:

        assert len(loader) > 0
        images, labels = next(iter(loader))
        assert images.shape == (4, 3, 224, 224)
        assert labels.shape == (4,)

        # Check DataLoader properties
        assert isinstance(loader, DataLoader), "Should return a DataLoader instance"

        # Check correct batch sizes:
        assert (
            loader.batch_size == 4
        ), f"Batch size should be 4, got {loader.batch_size}"


@pytest.mark.order(4)
def test_oxford_pet_dataset():
    """Test loading and configuration of OxfordIIITPet dataset.

    Tests that the OxfordIIITPet dataset:
    1. Loads correctly with binary classification (cats vs dogs)
    2. Has correct number of classes (2)
    3. Has proper dataset splits
    4. Returns correct data shapes
    """
    # Setup configuration for OxfordIIITPet
    TEST_DATA_DIR.mkdir(exist_ok=True)

    config = {
        "data": {
            "name": "OxfordIIITPet",
            "path": str(TEST_DATA_DIR),
            "batch_size": 8,
        }
    }

    data_manager = DataManager(config)

    # Test loading training data
    data_manager.load_train_data()

    assert (
        data_manager.num_classes == 2
    ), f"OxfordIIITPet binary should have 2 classes, got {data_manager.num_classes}"
    assert data_manager.train_dataset is not None, "Training dataset should be loaded"
    assert isinstance(
        data_manager.train_dataset, datasets.OxfordIIITPet
    ), "Should be OxfordIIITPet dataset"

    # Test loading validation data
    data_manager.load_val_data()
    assert data_manager.val_dataset is not None, "Validation dataset should be loaded"

    # Test data loaders
    train_loader = data_manager.get_train_loader()
    val_loader = data_manager.get_val_loader()

    # Check a batch from train loader
    images, labels = next(iter(train_loader))
    assert images.shape == (
        8,
        3,
        224,
        224,
    ), f"Expected shape (8, 3, 224, 224), got {images.shape}"
    assert labels.shape == (8,), f"Expected labels shape (8,), got {labels.shape}"
    assert labels.max() <= 1, "Binary classification should have labels 0 or 1"
    assert labels.min() >= 0, "Labels should be non-negative"


@pytest.mark.order(5)
def test_pcam_dataset():
    """Test loading and configuration of PCAM dataset.

    Tests that the PCAM (PatchCamelyon) dataset:
    1. Loads correctly for histopathology binary classification
    2. Has correct number of classes (2)
    3. Has proper train/val splits
    4. Returns correct data shapes for 96x96 patches
    """
    # Setup configuration for PCAM
    TEST_DATA_DIR.mkdir(exist_ok=True)

    config = {"data": {"name": "PCAM", "path": str(TEST_DATA_DIR), "batch_size": 16}}

    data_manager = DataManager(config)

    data_manager.num_workers = 0  # To avoid problems when loading the dataset

    # Test loading training data
    data_manager.load_train_data()

    assert (
        data_manager.num_classes == 2
    ), f"PCAM should have 2 classes (normal/tumor), got {data_manager.num_classes}"
    assert data_manager.train_dataset is not None, "Training dataset should be loaded"
    assert isinstance(
        data_manager.train_dataset, datasets.PCAM
    ), "Should be PCAM dataset"

    # Test loading validation data
    data_manager.load_val_data()
    assert data_manager.val_dataset is not None, "Validation dataset should be loaded"
    assert isinstance(
        data_manager.val_dataset, datasets.PCAM
    ), "Validation should be PCAM dataset"

    # Test data loaders
    train_loader = data_manager.get_train_loader()
    val_loader = data_manager.get_val_loader()

    # Check a batch from train loader
    images, labels = next(iter(train_loader))
    assert images.shape == (
        16,
        3,
        224,
        224,
    ), f"Expected shape (16, 3, 224, 224) after resize, got {images.shape}"
    assert labels.shape == (16,), f"Expected labels shape (16,), got {labels.shape}"

    # Check that loader is shuffled for training
    assert isinstance(
        train_loader.sampler, RandomSampler
    ), "Training data should be shuffled"
    assert isinstance(
        val_loader.sampler, SequentialSampler
    ), "Validation data should not be shuffled"

    # Verify class names are set
    assert (
        len(data_manager.class_names) == 2
    ), "Should have 2 class names for binary classification"
