import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from torch.utils.data import Subset

from src.visualizer import FeatureVisualizer
from src.data_manager import DataManager
from src.model_handler import ModelHandler

# Define a constant for the test data directory
TEST_DATA_DIR = Path("./test_data_cache")


@pytest.fixture
def visualizer_with_subset():
    """Provides a FeatureVisualizer instance configured with a small
      subset of data for fast testing.

    Creates a FeatureVisualizer with OxfordIIITPet dataset limited to 100 samples
    to ensure fast test execution. The fixture pre-configures both the DataManager
    with a subset and the ModelHandler with a pre-trained AlexNet model.

    Returns:
        FeatureVisualizer: Configured visualizer instance
        with subset data and loaded model.
    """
    TEST_DATA_DIR.mkdir(exist_ok=True)
    config = {
        "data": {"name": "OxfordIIITPet", "path": str(TEST_DATA_DIR), "batch_size": 4},
        "model": {"name": "alexnet"},
    }

    # Setup DataManager and load data
    dm = DataManager(config)
    dm.load_train_data()

    # Create subset for efficient testing
    subset_indices = list(range(min(100, len(dm.train_dataset))))
    dm.train_dataset = Subset(dm.train_dataset, subset_indices)

    # Setup ModelHandler with pre-trained weights
    mh = ModelHandler(config)
    mh.load_model()

    # Create FeatureVisualizer with pre-configured components
    visualizer = FeatureVisualizer(config)
    visualizer.data_manager = dm
    visualizer.model_handler = mh
    return visualizer


@patch("matplotlib.pyplot.show")
def test_feature_visualizer_run_automated(mock_show, visualizer_with_subset):
    """Tests the feature extraction and PCA visualization pipeline
      with automated validation.

    Verifies that the FeatureVisualizer can successfully extract features from a subset
    of training data, perform PCA dimensionality reduction, and prepare visualization
    data without displaying plots. Uses mocking to avoid GUI dependencies.

    Args:
        mock_show: Mocked matplotlib.pyplot.show to prevent plot display.
        visualizer_with_subset: Fixture providing configured FeatureVisualizer
        with subset data.
    """
    visualizer = visualizer_with_subset
    train_loader = visualizer.data_manager.get_train_loader()

    # Mock plotting method to capture results without displaying
    visualizer._plot_pca = MagicMock()

    # Execute feature extraction and PCA pipeline
    try:
        visualizer._extract_features_and_plot(train_loader)
    except Exception as e:
        pytest.fail(f"visualizer._extract_features_and_plot() raised an exception: {e}")

    # Verify plotting method was called and validate output dimensions
    visualizer._plot_pca.assert_called_once()
    args, _ = visualizer._plot_pca.call_args
    features_2d, labels = args

    expected_samples = len(visualizer.data_manager.train_dataset)

    assert features_2d.shape == (expected_samples, 2)
    assert len(labels) == expected_samples


def test_feature_visualizer_run_and_show_plot(visualizer_with_subset):
    """
    Runs the full visualizer pipeline on a small subset and displays the plot.
    """
    # 1. Get the visualizer instance from the fixture
    visualizer = visualizer_with_subset
    train_loader = visualizer.data_manager.get_train_loader()

    # 2. Execute the main method, which will now show the plot
    try:
        print("\nRunning visualizer test with a subset... a plot window should appear.")
        visualizer._extract_features_and_plot(train_loader)
        print("Plot window closed, test finished.")
    except Exception as e:
        pytest.fail(f"visualizer raised an exception during visual test: {e}")


@pytest.fixture
def pcam_visualizer_with_subset():
    """Provides a FeatureVisualizer instance configured with PCAM dataset subset
      for histopathology visualization.

    Creates a FeatureVisualizer with PCAM (PatchCamelyon) dataset limited to 200
    samples for efficient testing of histopathology patch classification.
    The PCAM dataset contains 96x96 pixel patches from lymph node sections for
    binary classification (normal vs tumor tissue).

    Returns:
        FeatureVisualizer: Configured visualizer instance with PCAM
        subset data and loaded model.
    """
    TEST_DATA_DIR.mkdir(exist_ok=True)
    config = {
        "data": {"name": "PCAM", "path": str(TEST_DATA_DIR), "batch_size": 8},
        "model": {"name": "resnet18"},  # Using ResNet18 for better
        # feature extraction on medical images
    }

    # Setup DataManager and load PCAM data
    dm = DataManager(config)
    dm.num_workers = 0  # Avoid multiprocessing issues in tests
    dm.load_train_data()

    # Create subset for efficient testing (200 samples for good class representation)
    subset_indices = list(range(min(200, len(dm.train_dataset))))
    dm.train_dataset = Subset(dm.train_dataset, subset_indices)

    # Setup ModelHandler with pre-trained ResNet18
    mh = ModelHandler(config)
    mh.load_model()

    # Create FeatureVisualizer with pre-configured components
    visualizer = FeatureVisualizer(config)
    visualizer.data_manager = dm
    visualizer.model_handler = mh
    return visualizer


def test_pcam_feature_visualizer_with_plot(pcam_visualizer_with_subset):
    """Tests feature extraction and visualization on PCAM dataset with actual plot.

    Runs the complete visualization pipeline on PCAM histopathology patches and displays
    the resulting PCA plot. This test is useful for manual inspection of feature
    clustering patterns between normal and tumor tissue patches.

    The plot will show:
    - Principal Component 1 vs Principal Component 2
    - Two classes: normal tissue (class 0) vs tumor tissue (class 1)
    - Feature separability in the reduced dimensional space

    Args:
        pcam_visualizer_with_subset: Fixture providing configured FeatureVisualizer
        with PCAM subset data.
    """
    visualizer = pcam_visualizer_with_subset
    train_loader = visualizer.data_manager.get_train_loader()

    # Execute the visualization pipeline with actual plot display
    try:
        print(f"\nRunning PCAM histopathology visualization test...")
        print(f"Dataset: {visualizer.data_manager.dataset_name}")
        print(f"Model: {visualizer.config['model']['name']}")
        print(f"Samples: {len(visualizer.data_manager.train_dataset)}")
        print(f"Classes: Normal tissue (0) vs Tumor tissue (1)")
        print(
            "A plot window should appear showing PCA visualization"
            " of histopathology features..."
        )

        visualizer._extract_features_and_plot(train_loader)
        print("PCAM visualization plot window closed, test finished.")
    except Exception as e:
        pytest.fail(f"PCAM visualizer raised an exception during visual test: {e}")
