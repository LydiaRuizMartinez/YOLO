from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from typing import Any


class DataManager:
    """Manages data loading flexibly using an internal configuration registry.

    This class provides a flexible way to load different datasets by maintaining
    a configuration registry for supported datasets and their parameters. It handles
    dataset instantiation, transformation pipeline creation, and DataLoader setup
    for both training and validation data.

    Attributes:
        DATASET_CONFIGS (dict): A dictionary mapping dataset names to their
        configuration dictionaries. Each configuration contains the dataset class,
        number of classes, and arguments for training and validation splits.
        config (dict): Configuration dictionary containing data specifications passed
        during initialization.
        dataset_name (str): Name of the selected dataset from the configuration.
        dataset_config (dict): Configuration dictionary for the specific dataset being
        used.
        train_dataset (torch.utils.data.Dataset or None): Training dataset instance.
        val_dataset (torch.utils.data.Dataset or None): Validation dataset instance.
        num_classes (int): Number of classes in the dataset.
        class_names (list[str]): List of class names from the dataset.
        num_workers (int): Number of workers for data loading processes.

    Example:
        >>> config = {
        ...     "data": {
        ...         "name": "CIFAR10",
        ...         "path": "./data",
        ...         "batch_size": 32
        ...     }
        ... }
        >>> data_manager = DataManager(config)
        >>> data_manager.load_data()
        >>> train_loader = data_manager.get_train_loader()
    """

    # TODO: Complete the dataset configurations based on each dataset's properties
    # (number of classes, splits, target types, etc.)

    DATASET_CONFIGS = {
        "CIFAR10": {
            "dataset_class": datasets.CIFAR10,
            "num_classes": 10,
            "train_args": {"train": True, "download": True},
            "val_args": {"train": False, "download": True},
        },
        "OxfordIIITPet": {
            "dataset_class": datasets.OxfordIIITPet,
            "num_classes": 2,
            "train_args": {
                "split": "trainval",
                "target_types": "binary-category",
                "download": True,
            },
            "val_args": {
                "split": "test",
                "target_types": "binary-category",
                "download": True,
            },
        },
        "PCAM": {
            "dataset_class": datasets.PCAM,
            "num_classes": 2,
            "train_args": {
                "split": "train",
                "download": False,
            },  # Note: PCAM has to be downloaded manually.
            "val_args": {"split": "val", "download": False},
        },
    }

    def __init__(self, config: dict) -> None:
        """Initialize the DataManager with configuration settings.

        Validates the provided configuration to ensure a supported dataset is specified,
        then caches the dataset configuration and initializes instance attributes.

        Args:
            config (dict): Configuration dictionary that must contain a 'data' key
                with 'name', 'path', and 'batch_size' subkeys. The dataset name
                must be present in the DATASET_CONFIGS registry.

        Raises:
            ValueError: If no dataset name is specified in config['data']['name'].
            ValueError: If the specified dataset is not supported (not found in
                DATASET_CONFIGS registry).

        Example:
            >>> config = {
            ...     "data": {
            ...         "name": "CIFAR10",
            ...         "path": "./datasets",
            ...         "batch_size": 64
            ...     }
            ... }
            >>> data_manager = DataManager(config)
        """
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.class_names: list[str] = []
        self.num_workers = 2

        # Validate and cache dataset configuration early
        data_config = self.config.get("data", {})
        dataset_name = data_config.get("name")

        # Cache the dataset configuration for later use
        self.dataset_name = dataset_name
        self.dataset_config = self.DATASET_CONFIGS[dataset_name]
        self.num_classes = self.dataset_config["num_classes"]

    def _get_transforms(self) -> Any:
        """Create and return a standardized data transformation pipeline.

        Creates a transforms.Compose pipeline optimized for transfer learning with
        ImageNet pre-trained models. The pipeline includes image resizing to ImageNet
        dimensions, tensor conversion, and normalization using ImageNet statistics.

        Returns:
            transforms.Compose: The composed transformation pipeline that includes:
                - Resize to images for ImageNet compatibility
                - Convert PIL images to tensors
                - Normalize with ImageNet statistics

        Note:
            These transformations are essential for transfer learning as they ensure
            input data matches the preprocessing used during ImageNet pre-training.

        Example:
            >>> data_manager = DataManager(config)
            >>> transforms = data_manager._get_transforms()
            >>> # transforms can now be applied to PIL images
        """
        # TODO
        imagenet_size = int(
            self.config.get("data", {}).get("image_size", 224)
        )  # Default input size for ImageNet pre-trained models is 224x224

        # Default mean and std values used for normalization in ImageNet-trained models
        mean = self.config.get("data", {}).get("mean", [0.485, 0.456, 0.406])
        std = self.config.get("data", {}).get("std", [0.229, 0.224, 0.225])

        return transforms.Compose(
            [
                transforms.Resize((imagenet_size, imagenet_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def load_train_data(self) -> None:
        """Load the training dataset using the dataset configuration.

        Uses the dataset class and parameters from the configuration registry
        to instantiate the training dataset. Applies the transformations from
        _get_transforms() and updates class information.

        Updates:
            self.train_dataset: The loaded training dataset instance
            self.class_names: List of class names from the dataset
        """
        # TODO
        ds_cls = self.dataset_config["dataset_class"]
        root = self.config.get("data", {}).get("path", "./data")

        # Apply the predefined transformation pipeline
        transform = self._get_transforms()

        args = dict(self.dataset_config["train_args"])
        self.train_dataset = ds_cls(
            root=root, transform=transform, target_transform=None, **args
        )

        # If the dataset provides a list of class names (e.g., CIFAR10), use it
        if hasattr(self.train_dataset, "classes") and isinstance(
            self.train_dataset.classes, (list, tuple)
        ):
            self.class_names = list(self.train_dataset.classes)
        # Otherwise, create a generic list of class labels as strings (e.g., ["0", "1"])
        else:
            self.class_names = [str(i) for i in range(self.num_classes)]

    def load_val_data(self) -> None:
        """Load the training dataset using the dataset configuration.

        Uses the dataset class and parameters from the configuration registry
        to instantiate the validation dataset. Applies the transformations from
        _get_transforms() and updates class information.

        Updates:
            self.val_dataset: The loaded training dataset instance
            self.class_names: List of class names from the dataset
        """
        # TODO
        ds_cls = self.dataset_config["dataset_class"]
        root = self.config.get("data", {}).get("path", "./data")

        # Apply the predefined transformation pipeline
        transform = self._get_transforms()

        args = dict(self.dataset_config["val_args"])
        self.val_dataset = ds_cls(
            root=root, transform=transform, target_transform=None, **args
        )

        if not self.class_names:
            # If the dataset provides a list of class names (e.g., CIFAR10), use it
            if hasattr(self.val_dataset, "classes") and isinstance(
                self.val_dataset.classes, (list, tuple)
            ):
                self.class_names = list(self.val_dataset.classes)
            # Otherwise, create a generic list of class labels as strings
            else:
                self.class_names = [str(i) for i in range(self.num_classes)]

    def load_data(self) -> None:
        """Loads both training and validation datasets.

        Convenience method that loads both training and validation data.
        """
        # TODO
        self.load_train_data()
        self.load_val_data()

    def get_train_loader(self) -> DataLoader:
        """Return a DataLoader for the training dataset.

        Creates a DataLoader with appropriate batch size and shuffling settings
        for training. Ensures the training dataset has been loaded before creating
        the loader.

        Returns:
            DataLoader: Training data loader configured for training

        Raises:
            ValueError: If training dataset has not been loaded yet
        """
        # TODO
        data_cfg = self.config.get("data", {})

        # Create a DataLoader for the training dataset
        return DataLoader(
            self.train_dataset,
            batch_size=int(data_cfg.get("batch_size", 32)),
            shuffle=bool(data_cfg.get("shuffle_train", True)),
            num_workers=self.num_workers,
            pin_memory=bool(data_cfg.get("pin_memory", True)),
            drop_last=bool(data_cfg.get("drop_last_train", False)),
        )

    def get_val_loader(self) -> DataLoader:
        """Return a DataLoader for the validation dataset.

        Creates a DataLoader with appropriate batch size and shuffling settings
        for validation. Ensures the validation dataset has been loaded before creating
        the loader.

        Returns:
            DataLoader: Validation data loader configured for validation

        Raises:
            ValueError: If validation dataset has not been loaded yet
        """
        # TODO
        data_cfg = self.config.get("data", {})

        # Create a DataLoader for the validation dataset
        return DataLoader(
            self.val_dataset,
            batch_size=int(data_cfg.get("batch_size", 32)),
            shuffle=bool(data_cfg.get("shuffle_val", False)),
            num_workers=self.num_workers,
            pin_memory=bool(data_cfg.get("pin_memory", True)),
            drop_last=bool(data_cfg.get("drop_last_val", False)),
        )
