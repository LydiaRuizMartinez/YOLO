"""
Script to define the architecture for the model.
"""

import torch
from PIL import Image, ImageDraw
from PIL.JpegImagePlugin import JpegImageFile
from torch import nn
from torchvision import transforms

from src.constants import MODEL, TARGET_SIZE_IMG, B, C, S
from src.utils import nms, to_image_coords


class CNNBlock(nn.Module):
    """
    Class to build a CNN block consisting of Conv + Batch Norm + LeakyReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        """
        Constructor of the class.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size.
            stride: Stride.
            padding: Padding.
        """

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, in_channels, height, width].

        Returns:
            Output tensor. Dimensions: [batch, out_channels, h_out, w_out] (h_out and
            w_out can be checked here:
            https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).
        """

        return self.leaky_relu(self.batchnorm(self.conv(x)))


class YOLO(nn.Module):
    """
    Class to build the YOLO architecture.
    """

    def __init__(self, in_channels: int = 3) -> None:
        """
        Constructor of the class.

        Args:
            in_channels: First input channels for the convolutional blocks.
        """

        super().__init__()

        self.cnn = self._create_cnn(MODEL, in_channels)
        self.mlp = self._create_mlp()

    def _create_cnn(
        self, architecture: list[tuple[int, int, int, int] | str], in_channels: int
    ) -> nn.Module:
        """
        Creates the convolutional layers.

        Args:
            architecture: Architecture of the convolutional layers. It is explained
                in the constants.py file.
            in_channels: Input channels for the first convolution.

        Returns:
            Architecture of the convolutional layers.
        """

        # TODO
        layers: list[nn.Module] = []

        for layer in architecture:
            if isinstance(layer, str):
                # Max-pooling marker (e.g., "M")
                if layer.upper() == "M":
                    layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
                else:
                    raise ValueError(f"Unknown layer string in MODEL: {layer}")

            # Single convolutional block: (kernel, out_channels, stride, padding)
            elif isinstance(layer, tuple):
                k, out_c, s, p = layer
                layers.append(CNNBlock(in_channels, out_c, k, s, p))
                in_channels = out_c

            # Repeated pattern: [conv1_tuple, conv2_tuple, num_repeats]
            elif isinstance(layer, list):
                if len(layer) != 3:
                    raise ValueError(
                        "Repeated block must be [conv1_tuple, conv2_tuple, num_repeats]"
                    )
                conv1, conv2, num_repeats = layer
                for _ in range(int(num_repeats)):
                    # First conv in the pair
                    k1, c1, s1, p1 = conv1
                    layers.append(CNNBlock(in_channels, c1, k1, s1, p1))
                    in_channels = c1
                    # Second conv in the pair
                    k2, c2, s2, p2 = conv2
                    layers.append(CNNBlock(in_channels, c2, k2, s2, p2))
                    in_channels = c2
            else:
                raise TypeError(f"Unsupported layer type in MODEL: {type(layer)}")

        # Wrap the collected blocks into a single sequential module
        return nn.Sequential(*layers)

    def _create_mlp(self) -> nn.Module:
        """
        Creates the fully connected layers. You have the freedom to define the number
        and type of layers (for example, linear, dropout, leaky relu...) as you wish.

        Returns:
            Architecture of the fully connected layers.
        """

        # TODO
        output_dim = S * S * (C + 5 * B)
        return nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(4096),  # infers in_features on first call
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, output_dim),  # final logits for all cells/boxes/classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, channels, height, width].

        Returns:
            Output tensor: Dimensions: [batch, S, S, 5 * B + C].
        """

        # TODO
        x = self.cnn(x)  # [batch, C*, H*, W*]
        x = self.mlp(x)  # [batch, S*S*(C + 5*B)]
        x = x.reshape(-1, S, S, 5 * B + C)  # [batch, S, S, C + 5*B]
        return x

    def predict(self, img_path: str) -> list[torch.Tensor]:
        """
        Prediction given an image.

        Args:
            img_path: Path to the image (jpg, png...).

        Returns:
            Bounding boxes after NMS. Each one has dimension C + 5 and the last 4
            elements indicate the coordinates of the box.
        """

        self.eval()

        x = transforms.ToTensor()(
            Image.open(img_path).resize(TARGET_SIZE_IMG).convert("RGB")
        )

        predicted_boxes = self(x.unsqueeze(0)).squeeze(0)
        scaled_boxes = to_image_coords(
            predicted_boxes, TARGET_SIZE_IMG[0], TARGET_SIZE_IMG[1]
        )

        return nms(scaled_boxes)

    def draw_predictions(self, img_path: str, output_path: str = "") -> JpegImageFile:
        """
        Draws the predicted bounding boxes of the image.

        Args:
            img_path: Path to the image.
            output_path: Path were the image created is saved (only if a path is given).

        Returns:
            Image with the bounding boxes.
        """

        img = Image.open(img_path).resize(TARGET_SIZE_IMG).convert("RGB")
        output_img = ImageDraw.Draw(img)
        predicted_boxes = self.predict(img_path)

        for predicted_box in predicted_boxes:
            c_x, c_y, width, height = predicted_box[C + 1 :]
            # Change coordinates to (x1, y1, x2, y2)
            try:
                output_img.rectangle(
                    [
                        float(c_x - width / 2),
                        float(c_y - height / 2),
                        float(c_x + width / 2),
                        float(c_y + height / 2),
                    ],
                    outline="red",
                    width=3,
                )
            except ValueError:  # wrong coordinates (for example, y1 < y0)
                pass

        if output_path:
            img.save(output_path)

        return img  # type: ignore
