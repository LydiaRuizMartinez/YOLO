# src/utils.py

from torch import nn


def set_nested_attr(obj: nn.Module, attr_string: str, value: nn.Module) -> None:
    """Set a nested attribute on a PyTorch module using dot notation.

    This helper function enables setting deeply nested attributes on PyTorch modules
    using a string path with dot notation (e.g., 'classifier.6' or 'features.conv1').
    It handles both regular attribute access and Sequential module indexing, making
    it particularly useful for modifying specific layers in complex model architectures.

    The function intelligently handles two common PyTorch patterns:
    1. Regular attribute access (e.g., model.fc, model.classifier)
    2. Sequential module indexing (e.g., model.classifier[6])

    Args:
        obj (nn.Module): The root PyTorch module object on which to set the attribute.
            This is typically a complete model or a major component of a model.
        attr_string (str): Dot-separated string specifying the path to the target
            attribute. Examples include 'fc', 'classifier.6', 'features.conv1.weight'.
            Numeric components are treated as Sequential module indices.
        value (nn.Module): The new module to assign to the specified attribute path.
            This is typically a replacement layer (e.g., nn.Linear, nn.Conv2d).

    Raises:
        AttributeError: If any intermediate attribute in the path doesn't exist
            on the parent object.
        ValueError: If a numeric index is provided for a non-Sequential module.
        IndexError: If a numeric index is out of bounds for a Sequential module.

    Examples:
        Replace a simple final classifier layer:
        >>> model = models.resnet18()
        >>> new_fc = nn.Linear(512, 10)
        >>> set_nested_attr(model, 'fc', new_fc)

        Replace a layer within a Sequential module (VGG-style):
        >>> model = models.vgg16()
        >>> new_classifier = nn.Linear(4096, 5)
        >>> set_nested_attr(model, 'classifier.6', new_classifier)

        Replace a convolutional layer in features:
        >>> model = models.alexnet()
        >>> new_conv = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        >>> set_nested_attr(model, 'features.3', new_conv)

    Note:
        This function modifies the original module in-place. The numeric indices
        in the attribute string are automatically converted to integers for
        Sequential module access. The function assumes that numeric components
        in the path refer to Sequential module indices.
    """
    attrs = attr_string.split(".")
    parent = obj
    for attr in attrs[:-1]:
        parent = getattr(parent, attr)
    final_attr = attrs[-1]
    # Handle cases where the parent is a Sequential module and final attribute an index
    if final_attr.isdigit() and isinstance(parent, nn.Sequential):
        parent[int(final_attr)] = value
    else:
        setattr(parent, final_attr, value)


def get_final_layer(model: nn.Module) -> tuple[str, nn.Linear]:
    """
    Retrieves the name and the final fully-connected layer of a PyTorch model.
    This iterates backwards through all named modules
    to find the *last nn.Linear layer*, ignoring final activation or dropout layers.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        tuple: A tuple containing the layer's name (e.g., 'fc' or 'classifier.6')
        and the layer module itself.

    Raises:
        ValueError: If no nn.Linear layer is found in the model.
    """
    # TODO
    # Use model.named_modules() to iterate through all modules
    all_modules = list(model.named_modules())

    # Iterate backwards through all modules to find the last Linear layer
    for name, module in reversed(all_modules):
        if isinstance(module, nn.Linear):
            return name, module
