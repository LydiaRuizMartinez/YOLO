"""
Creates a Pytorch dataset to load the dataset used.
"""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.constants import B, C, S, TARGET_SIZE_IMG


class RaccoonDataset(Dataset):
    """
    Adapted Dataset class for our dataset.
    """

    def __init__(
        self,
        img_dir: str,
        label_dir: str,
    ) -> None:
        """
        Constructor of the class.

        Args:
            img_dir: Path to the directory of the images.
            label_dir: Path to the directory of the labels.
        """

        self.imgs_dir = img_dir
        self.labels_dir = label_dir

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            Length of the dataset.
        """

        return len(os.listdir(self.labels_dir))

    def _generate_label(self, boxes: list[list[int | float]]) -> torch.Tensor:
        """
        Generates the labels with coordinates respect to the cells, not the hole image.
        We will assume at most we can have one object per cell. Therefore, the
        dimensions of the generated label will be [S, S, 5 * B2 + C]. If we wanted to
        detect at most n objects the dimensions would be [S, S, n, 5 * B + C].

        Args:
            boxes: Labels with respect to the hole image. Each label has 5 elements:
                class, center_x, center_y, width and height.

        Returns:
            Labels with respect to the cells. Dimensions: [S, S, 5 * B + C].
        """

        # TODO
        label = torch.zeros((S, S, 5 * B + C), dtype=torch.float32)

        eps = 1e-6
        for cls, cx, cy, w, h in boxes:
            # Clamp bounding box values to valid ranges
            cx = float(min(max(cx, 0.0), 1.0 - eps))
            cy = float(min(max(cy, 0.0), 1.0 - eps))
            w = float(min(max(w, 0.0), 1.0))
            h = float(min(max(h, 0.0), 1.0))

            # Compute grid cell indices (row i for y, column j for x)
            i = int(S * cy)
            j = int(S * cx)

            # Compute coordinates relative to the cell
            x_cell = S * cx - j
            y_cell = S * cy - i
            w_cell = w * S
            h_cell = h * S

            # One-hot encode the class in the first C channels
            cls_idx = int(cls)
            if 0 <= cls_idx < C:
                label[i, j, cls_idx] = 1.0

            # Find an available bounding box slot (among B slots in the cell)
            slot_k = None
            for k in range(B):
                obj_idx = C + 5 * k
                if label[i, j, obj_idx].item() == 0.0:
                    slot_k = k
                    break
            if slot_k is None:
                continue

            # Fill in objectness and bounding box values for the selected slot
            base = C + 5 * slot_k
            label[i, j, base] = 1.0
            label[i, j, base + 1 : base + 5] = torch.tensor(
                [x_cell, y_cell, w_cell, h_cell], dtype=torch.float32
            )

        return label

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the image and its labels.

        Args:
            index: Index of the image and labels.

        Returns:
            The image and its labels. Dimensions of the image: [channels, height,
            width]. Dimensions of the labels: [S, S, 5 * B + C].
        """

        # TODO
        candidates = []
        for stem in (f"raccoon-{index}", f"{index}", f"raccoon_{index}"):
            for ext in (".jpg", ".jpeg", ".png"):
                candidates.append(os.path.join(self.imgs_dir, stem + ext))

        # Pick the first existing image path among the candidates (if any).
        img_path = next((p for p in candidates if os.path.exists(p)), None)

        if img_path is not None:
            # Load the image in RGB and apply basic transforms
            img = Image.open(img_path).convert("RGB")
            transform = transforms.Compose(
                [
                    transforms.Resize(TARGET_SIZE_IMG),
                    transforms.ToTensor(),
                ]
            )
            x = transform(img)
        else:
            x = torch.zeros(
                (3, TARGET_SIZE_IMG[0], TARGET_SIZE_IMG[1]), dtype=torch.float32
            )

        # Build possible label filenames matching the same index/stems.
        label_candidates = [
            os.path.join(self.labels_dir, f"{index}.txt"),
            os.path.join(self.labels_dir, f"raccoon-{index}.txt"),
            os.path.join(self.labels_dir, f"raccoon_{index}.txt"),
        ]

        # Pick the first existing label file (if any).
        label_path = next((p for p in label_candidates if os.path.exists(p)), None)

        # Parse YOLO-like label lines: "class cx cy w h"
        boxes: list[list[int | float]] = []
        if label_path is not None:
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls = int(float(parts[0]))
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    boxes.append([cls, cx, cy, w, h])

        y = self._generate_label(boxes)
        return x, y
