# src/model_handler.py
from torch import nn
from torchvision import models
from src.utils import get_final_layer, set_nested_attr


class ModelHandler:
    """Handles model loading and modification for fine-tuning.

    This class provides functionality to load pre-trained models from torchvision
    and dynamically modify them for fine-tuning on custom datasets. It automatically
    finds and replaces the final classification layer to match the target number of
    classes.

    Attributes:
        MODEL_REGISTRY (dict): A dictionary mapping model names to their corresponding
            torchvision model classes. Currently supports resnet18, vgg16, and alexnet.
        config (dict): Configuration dictionary containing model specifications.
        model (torch.nn.Module or None): The loaded PyTorch model instance.

    Example:
        >>> config = {"model": {"name": "resnet18"}}
        >>> handler = ModelHandler(config)
        >>> handler.load_model()
        >>> handler.prepare_for_finetuning(num_classes=10)
    """

    # TODO
    MODEL_REGISTRY = {
        "resnet18": (models.resnet18, getattr(models, "ResNet18_Weights", None)),
        "vgg16": (models.vgg16, getattr(models, "VGG16_Weights", None)),
        "alexnet": (models.alexnet, getattr(models, "AlexNet_Weights", None)),
        "inception": (
            models.inception_v3,
            getattr(models, "Inception_V3_Weights", None),
        ),
        "inception_v3": (
            models.inception_v3,
            getattr(models, "Inception_V3_Weights", None),
        ),
    }

    def __init__(self, config: dict[str, dict[str, str]]) -> None:
        """Initialize the ModelHandler with configuration settings.

        Args:
            config (dict): Configuration dictionary that must contain a 'model' key
                with a 'name' subkey specifying which model to use. The model name
                must be present in the MODEL_REGISTRY.

        Example:
            >>> config = {
            ...     "model": {
            ...         "name": "resnet18"
            ...     }
            ... }
            >>> handler = ModelHandler(config)
        """
        self.config = config
        self.model = None

    def load_model(self) -> None:
        """Load a pre-trained model from torchvision based on configuration.

        Loads the specified model with ImageNet pre-trained weights. The model name
        is retrieved from the configuration and must exist in the MODEL_REGISTRY.

        Raises:
            ValueError: If the specified model name is not found in MODEL_REGISTRY.

        Note:
            This method loads models with 'IMAGENET1K_V1' weights, which are the
            standard ImageNet pre-trained weights from torchvision.

        Example:
            >>> handler = ModelHandler({"model": {"name": "resnet18"}})
            >>> handler.load_model()
            Loading pre-trained model: resnet18
        """
        # TODO
        model_name = self.config["model"]["name"]

        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Model '{model_name}' not found in internal MODEL_REGISTRY. "
                f"Available models: {list(self.MODEL_REGISTRY.keys())}"
            )

        model_class, weights_class = self.MODEL_REGISTRY[model_name]

        # Load the model with pre-trained weights
        if weights_class is not None:
            self.model = model_class(weights=weights_class.DEFAULT)
        else:
            self.model = model_class(pretrained=True)

    def replace_last_layer(self, num_classes: int) -> None:
        """Prepare the loaded model for fine-tuning by replacing the final classifier.

        This method dynamically finds the model's final classification layer and
        replaces it with a new linear layer that outputs the specified number of
        classes.
        This is essential for transfer learning when the target dataset has a different
        number of classes than the original pre-trained model
        (e.g., ImageNet's 1000 classes).

        Args:
            num_classes (int): The number of output classes for the new classification
                layer. Must be a positive integer.

        Raises:
            RuntimeError: If no model has been loaded. Must call load_model() first.

        Note:
            The method preserves the input feature dimension of the original final
            layer while only changing the output dimension to match num_classes.
            This ensures compatibility with the pre-trained feature extractor layers.

        Example:
            >>> handler = ModelHandler({"model": {"name": "resnet18"}})
            >>> handler.load_model()
            >>> handler.replace_last_layer(num_classes=5)
            Model 'resnet18' adapted for fine-tuning with 5 classes.
            Replaced layer 'fc' with new classifier:
            Linear(in_features=512, out_features=5, bias=True)
        """
        # TODO

        layer_name, final_layer = get_final_layer(self.model)
        in_features = final_layer.in_features

        # Create a new linear layer with the same input features but new output classes
        new_classifier = nn.Linear(in_features, num_classes)

        # Replace the final layer using our utility function
        set_nested_attr(self.model, layer_name, new_classifier)
