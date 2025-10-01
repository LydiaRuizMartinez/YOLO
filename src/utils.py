"""
Script to write some necessary functions not related with the rest of the scripts.
"""

import os
from typing import Literal

import torch
from torch import nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.constants import EPSILON, TARGET_SIZE_IMG, B, C, S


def iou(
    boxes_preds: torch.Tensor,
    boxes_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates Intersection over Union for each pair of the batch.

    Args:
        boxes_preds: Predictions of bounding boxes, Dimensions: [batch_size, S, S, 4].
        boxes_labels: Correct bounding boxes.  Dimensions: [batch_size, S, S, 4].

    Returns:
        Intersection over Union for each pair of the batch. Dimensions:
        [batch_size, S, S].
    """

    # TODO
    # Convert predicted boxes from (cx, cy, w, h) to corner format (x1, y1, x2, y2)
    pred_x1 = boxes_preds[..., 0] - boxes_preds[..., 2] / 2
    pred_y1 = boxes_preds[..., 1] - boxes_preds[..., 3] / 2
    pred_x2 = boxes_preds[..., 0] + boxes_preds[..., 2] / 2
    pred_y2 = boxes_preds[..., 1] + boxes_preds[..., 3] / 2

    # Convert ground-truth boxes to corner format
    lab_x1 = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
    lab_y1 = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
    lab_x2 = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
    lab_y2 = boxes_labels[..., 1] + boxes_labels[..., 3] / 2

    # Compute intersection rectangle coordinates
    x1 = torch.maximum(pred_x1, lab_x1)
    y1 = torch.maximum(pred_y1, lab_y1)
    x2 = torch.minimum(pred_x2, lab_x2)
    y2 = torch.minimum(pred_y2, lab_y2)

    # Intersection width and height (clamped to avoid negatives)
    inter_w = torch.clamp(x2 - x1, min=0.0)
    inter_h = torch.clamp(y2 - y1, min=0.0)
    intersection = inter_w * inter_h

    # Areas of predictions and ground-truth boxes
    area_preds = torch.clamp(pred_x2 - pred_x1, min=0.0) * torch.clamp(
        pred_y2 - pred_y1, min=0.0
    )
    area_labels = torch.clamp(lab_x2 - lab_x1, min=0.0) * torch.clamp(
        lab_y2 - lab_y1, min=0.0
    )

    union = area_preds + area_labels - intersection + EPSILON
    iou_val = intersection / union

    # Return IoU with an extra last dimension
    return iou_val.unsqueeze(-1)  # [batch, S, S, 1]


def nms(
    predicted_boxes: list[torch.Tensor],
    threshold_confidence: float = 0.5,
    threshold_repeated: float = 0.5,
) -> list[torch.Tensor]:
    """
    Applies Non Max Suppression given the predicted boxes. When updating the list, if
    two boxes have different classes, we do not delete them even if their IoU is high.

    Parameters:
        predicted_boxes: List with the bounding boxes predicted. They have to be
            relative to the hole image, not to each cell. Each box has dimension C + 5.
        threshold_confidence: Threshold to remove predicted bounding boxes with low
            confidence.
        threshold_repeated: Threshold to remove predicted boxes because we consider they
            refer to the same box.

    Returns:
        Predicted bounding boxes after NMS. Each box has dimension C + 5.
    """

    # TODO
    if not predicted_boxes:
        return []

    boxes = []
    for b in predicted_boxes:
        # Ensure tensor, detached and float
        b = (
            b.clone().detach().float()
            if isinstance(b, torch.Tensor)
            else torch.tensor(b, dtype=torch.float32)
        )
        # Filter by confidence
        if b[1] >= threshold_confidence:
            boxes.append(b)

    # If all boxes were filtered out by confidence
    if not boxes:
        return []

    # Group boxes by (integer) class id.
    by_class: dict[int, list[torch.Tensor]] = {}
    for b in boxes:
        cls = int(b[0].item())
        by_class.setdefault(cls, []).append(b)

    kept: list[torch.Tensor] = []

    def corners(t: torch.Tensor) -> torch.Tensor:
        """
        Convert (cx, cy, w, h) to corner format (x1, y1, x2, y2).
        Returns a 1D tensor [x1, y1, x2, y2].
        """
        cx, cy, w, h = t[2], t[3], t[4], t[5]
        return torch.tensor(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=torch.float32
        )

    # Perform NMS independently for each class
    for cls, cls_boxes in by_class.items():
        cls_boxes = sorted(cls_boxes, key=lambda b: float(b[1]), reverse=True)
        while cls_boxes:
            # Keep the highest-confidence box
            best = cls_boxes.pop(0)
            kept.append(best)

            if not cls_boxes:
                break

            # Compute IoU of remaining boxes against the chosen "best" box
            best_c = corners(best).view(1, 1, 1, 4)
            remaining: list[torch.Tensor] = []

            for b in cls_boxes:
                b_c = corners(b).view(1, 1, 1, 4)
                # Intersection rectangle
                x1 = torch.maximum(best_c[..., 0], b_c[..., 0])
                y1 = torch.maximum(best_c[..., 1], b_c[..., 1])
                x2 = torch.minimum(best_c[..., 2], b_c[..., 2])
                y2 = torch.minimum(best_c[..., 3], b_c[..., 3])

                inter_w = torch.clamp(x2 - x1, min=0.0)
                inter_h = torch.clamp(y2 - y1, min=0.0)
                inter = inter_w * inter_h

                # Areas
                area_best = torch.clamp(
                    best_c[..., 2] - best_c[..., 0], min=0.0
                ) * torch.clamp(best_c[..., 3] - best_c[..., 1], min=0.0)
                area_b = torch.clamp(b_c[..., 2] - b_c[..., 0], min=0.0) * torch.clamp(
                    b_c[..., 3] - b_c[..., 1], min=0.0
                )

                # IoU = inter / (area_best + area_b - inter)
                union = area_best + area_b - inter + EPSILON
                iou_val = (inter / union).squeeze()

                if float(iou_val) <= threshold_repeated:
                    remaining.append(b)

            # Continue with the boxes that survived suppression
            cls_boxes = remaining

    return kept


def to_image_coords(boxes: torch.Tensor, img_w: int, img_h: int) -> list[torch.Tensor]:
    """
    Transforms the coordinates of the bounding boxes from relative to each cell to
    relative to the hole image.

    Args:
        boxes: Predicted bounding boxes. Dimensions: [S, S, 5 * B + C].
        img_w: Width of the image.
        img_h: Height of the image.

    Returns:
        List with the scaled bounding boxes to the hole image. Each tensor has dimension
        C + 5.
    """

    # TODO
    # Validate expected shape: [S, S, C + 5 * B]
    if (
        boxes.dim() != 3
        or boxes.shape[0] != S
        or boxes.shape[1] != S
        or boxes.shape[2] != (C + 5 * B)
    ):
        raise ValueError("Unexpected boxes shape, expected [S, S, C + 5 * B]")

    all_boxes: list[torch.Tensor] = []

    # Loop over each cell in the SxS grid
    for i in range(S):  # y direction
        for j in range(S):  # x direction
            # Get the class probability for this cell
            if C == 0:
                class_prob = torch.tensor(1.0, dtype=torch.float32)
            elif C == 1:
                class_prob = boxes[i, j, 0]
            else:
                class_prob = torch.max(boxes[i, j, :C])

            # For each bounding box slot in this cell
            for k in range(B):
                base = C + 5 * k
                obj = boxes[i, j, base + 0]
                x_cell = boxes[i, j, base + 1]
                y_cell = boxes[i, j, base + 2]
                w_cell = boxes[i, j, base + 3]
                h_cell = boxes[i, j, base + 4]

                # Convert from cell-relative to image-normalized [0,1] coordinates
                cx_norm = (j + x_cell) / S
                cy_norm = (i + y_cell) / S
                w_norm = w_cell / S
                h_norm = h_cell / S

                # Scale normalized coordinates to pixel values
                cx = cx_norm * img_w
                cy = cy_norm * img_h
                w = w_norm * img_w
                h = h_norm * img_h

                # Final box format: [class_conf, objectness, cx, cy, w, h]
                out = torch.tensor([class_prob, obj, cx, cy, w, h], dtype=torch.float32)
                all_boxes.append(out)

    return all_boxes


# This last part is just to calculate the mAP. It is a little bit annoying.


def mean_average_precision(subset: Literal["train", "test"], model: nn.Module) -> float:
    """
    Calculates mAP in the test set.

    Args:
        subset: Subset where the mAP is calculated.
        model: YOLO.

    Returns:
        Calculated mAP.
    """

    targets = _get_targets_map(subset)
    predictions = _get_predictions_map(subset, model)

    metric = MeanAveragePrecision(
        iou_type="bbox", iou_thresholds=[0.5], class_metrics=True
    )
    metric.update(predictions, targets)
    result = metric.compute()

    return float(result["map"])


def _get_targets_map(subset: Literal["train", "test"]) -> list[dict[str, torch.Tensor]]:
    """
    Gets the targets in the correct format for mAP calculation.

    Args:
        subset: Subset where the mAP is calculated.

    Returns:
        List of dictionaries with the form: {"boxes": ..., "labels": ...}.
    """

    targets = []

    for i in range(len(os.listdir(f"data/labels_resized/{subset}"))):
        temp_box = []
        temp_label = []
        with open(f"data/labels_resized/{subset}/{i}.txt", encoding="utf8") as f:
            for label in f.readlines():
                label, c_x, c_y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                temp_label.append(label)
                # Change coordinates to (x1, y1, x2, y2)
                temp_box.append(
                    torch.tensor(
                        [
                            (c_x - width / 2) * TARGET_SIZE_IMG[0],
                            (c_y - height / 2) * TARGET_SIZE_IMG[1],
                            (c_x + width / 2) * TARGET_SIZE_IMG[0],
                            (c_y + height / 2) * TARGET_SIZE_IMG[1],
                        ]
                    )
                )
        boxes = torch.stack(temp_box)
        labels = torch.tensor(temp_label)
        targets.append({"boxes": boxes, "labels": labels})

    return targets


def _get_predictions_map(
    subset: Literal["train", "test"], model: nn.Module
) -> list[dict[str, torch.Tensor]]:
    """
    Gets the predictions in the correct format for mAP calculation.

    Args:
        subset: Subset where the mAP is calculated.
        model: YOLO model.

    Returns:
        List of dictionaries with the form: {"boxes": ..., "scores": ...,
        "labels": ...}.
    """

    dir_path = f"data/images/{subset}"
    predictions = []

    for i in range(len(os.listdir(dir_path))):
        bounding_boxes = model.predict(f"{dir_path}/raccoon-{i}.jpg")
        boxes = []
        scores = []
        labels = []
        if len(bounding_boxes) > 0:
            for bounding_box in bounding_boxes:
                c_x, c_y, width, height = bounding_box[C + 1 :]
                # Change coordinates to (x1, y1, x2, y2)
                normalized_bb = torch.tensor(
                    [
                        float(c_x - width / 2),
                        float(c_y - height / 2),
                        float(c_x + width / 2),
                        float(c_y + height / 2),
                    ]
                )
                boxes.append(normalized_bb)
                scores.append(torch.min(torch.tensor(1.0), bounding_box[C]))
                labels.append(torch.argmax(bounding_box[:C]))
            predictions.append(
                {
                    "boxes": torch.stack(boxes),
                    "scores": torch.stack(scores),
                    "labels": torch.stack(labels),
                }
            )
        else:  # no predictions were made
            predictions.append(
                {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }
            )

    return predictions
