"""
Script to define the YOLO loss.
"""

import torch
from torch import nn

from src.constants import EPSILON, LAMBDA_COORD, LAMBDA_NOOBJ, C
from src.utils import iou


class YOLOLoss(nn.Module):
    """
    Implementation of YOLO loss function from the original paper. Some parts are
    hardcoded because we assume B = 2.
    """

    def __init__(self) -> None:
        """
        Constructor of the class.
        """

        super().__init__()

        self.mse = nn.MSELoss()
        # Bind constants used throughout the loss
        self.lambda_coord = LAMBDA_COORD
        self.lambda_noobj = LAMBDA_NOOBJ
        self.C = C
        self.eps = EPSILON

    def _coordinates_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        one_i_obj: torch.Tensor,
        best_box: torch.Tensor,
    ) -> torch.Tensor:
        """
        Obtains the loss for the predicted coordinates of the bounding boxes.

        Args:
            predictions: Predictions of the model. Dimensions: [batch, S, S, 5 * B + C].
            targets: Ground truths. Dimensions: [batch, S, S, 5 * B + C].
            one_i_obj: Binary tensor indicating if there is an object in each cell.
                Dimensions: [batch, S, S, 1].
            best_box: Binary tensor indicating the best box for each cell, which is the
                one with more confidence. For position (i, j): 0 -> first bounding box,
                1 -> second bounding box, because we assume B = 2). Dimensions:
                [batch_size, S, S, 1].

        Returns:
            Loss for the predicted coordinates. It is a scalar.
        """

        # TODO
        # Select the responsible predicted box per cell
        box_predictions = one_i_obj * (
            (
                best_box * predictions[..., self.C + 6 : self.C + 10]
                + (1 - best_box) * predictions[..., self.C + 1 : self.C + 5]
            )
        )

        # Ground-truth coordinates always come from the (single) target box per cell.
        box_targets = one_i_obj * targets[..., self.C + 1 : self.C + 5]

        # Apply the "sqrt trick" to width and height
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # Mean squared error over (x, y, sqrt(w), sqrt(h)) in object cells.
        coord_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        return self.lambda_coord * coord_loss

    def _object_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        one_i_obj: torch.Tensor,
        best_box: torch.Tensor,
    ) -> torch.Tensor:
        """
        Obtains the loss for the prediction of if a cell contains an object.

        Args:
            predictions: Predictions of the model. Dimensions: [batch, S, S, 5 * B + C].
            targets: Ground truths. Dimensions: [batch, S, S, 5 * B + C].
            one_i_obj: Binary tensor indicating if there is an object in each cell.
                Dimensions: [batch, S, S, 1].
            best_box: Binary tensor indicating the best box for each cell, which is the
                one with more confidence. For position (i, j): 0 -> first bounding box,
                1 -> second bounding box, because we assume B = 2). Dimensions:
                [batch_size, S, S, 1].

        Returns:
            Loss for the prediction of if a cell contains an object. It is a scalar.
        """

        # TODO
        # Select the objectness prediction from the responsible box
        pred_box = (
            best_box * predictions[..., self.C + 5 : self.C + 6]
            + (1 - best_box) * predictions[..., self.C : self.C + 1]
        )

        # Compute MSE only in cells that actually contain an object
        object_loss = self.mse(
            torch.flatten(one_i_obj * pred_box),
            torch.flatten(one_i_obj * targets[..., self.C : self.C + 1]),
        )

        return object_loss

    def _no_object_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        one_i_obj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Obtains the loss for the prediction of if a cell does not contain an object.

        Args:
            predictions: Predictions of the model. Dimensions: [batch, S, S, 5 * B + C].
            targets: Ground truths. Dimensions: [batch, S, S, 5 * B + C].
            one_i_obj: Binary tensor indicating if there is an object in each cell.
                Dimensions: [batch, S, S].

        Returns:
            Loss for the prediction of if a cell does not contain an object. It is a
            scalar.
        """

        # TODO
        # Inverse mask
        noobj = 1.0 - one_i_obj

        # Objectness predictions for the two bounding boxes in each cell
        conf_b1 = predictions[..., self.C : self.C + 1]
        conf_b2 = predictions[..., self.C + 5 : self.C + 6]

        # Ground-truth objectness
        tgt_conf = targets[..., self.C : self.C + 1]

        # Compute MSE for both box confidences
        loss_b1 = self.mse((noobj * conf_b1).flatten(), (noobj * tgt_conf).flatten())
        loss_b2 = self.mse((noobj * conf_b2).flatten(), (noobj * tgt_conf).flatten())

        return self.lambda_noobj * 0.5 * (loss_b1 + loss_b2)

    def _class_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        one_i_obj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Obtains the loss for the class of the objects.

        Args:
            predictions: Predictions of the model. Dimensions: [batch, S, S, 5 * B + C].
            targets: Ground truths. Dimensions: [batch, S, S, 5 * B + C].
            one_i_obj: Binary tensor indicating if there is an object in each cell.
                Dimensions: [batch, S, S, 1].

        Returns:
            Loss for the prediction of the class of the objects. It is a scalar.
        """

        # TODO
        # Select class predictions only for cells containing objects
        pred_cls = one_i_obj * predictions[..., : self.C]
        tgt_cls = one_i_obj * targets[..., : self.C]

        # Compute MSE between predicted class probabilities and one-hot targets
        return self.mse(
            pred_cls.flatten(end_dim=-2),
            tgt_cls.flatten(end_dim=-2),
        )

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            predictions: Predictions of the model. Dimensions: [batch, S, S, 5 * B + C].
            targets: Ground truths. Dimensions: [batch, S, S, 5 * B + C].

        Returns:
            YOLO loss and each of the sub-losses.
        """

        # TODO
        # Mask indicating which cells contain an object
        one_i_obj = targets[..., self.C : self.C + 1]

        # Compute IoU of predictions vs ground truth for each of the 2 bounding boxes
        iou_b1 = iou(
            predictions[..., self.C + 1 : self.C + 5],
            targets[..., self.C + 1 : self.C + 5],
        )
        iou_b2 = iou(
            predictions[..., self.C + 6 : self.C + 10],
            targets[..., self.C + 1 : self.C + 5],
        )

        # Ensure IoU tensors have the right shape
        if iou_b1.dim() == predictions.dim() - 1:  # [batch, S, S, 1]
            iou_b1 = iou_b1.unsqueeze(-1)
            iou_b2 = iou_b2.unsqueeze(-1)

        # Choose the "responsible" box
        best_box = (iou_b2 > iou_b1).float()

        # Compute the four components of YOLO loss
        coord_loss = self._coordinates_loss(predictions, targets, one_i_obj, best_box)
        object_loss = self._object_loss(predictions, targets, one_i_obj, best_box)
        no_object_loss = self._no_object_loss(predictions, targets, one_i_obj)
        class_loss = self._class_loss(predictions, targets, one_i_obj)

        # Sum all components into the total loss
        total = coord_loss + object_loss + no_object_loss + class_loss

        return coord_loss, object_loss, no_object_loss, class_loss, total
