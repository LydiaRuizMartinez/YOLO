"""
Script to train the model.
"""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from src.constants import BATCH_SIZE, EPOCHS, LR, PATH_MODEL
from src.data.dataset import RaccoonDataset
from src.model.loss import YOLOLoss
from src.model.model import YOLO
from src.utils import mean_average_precision

from typing import Any, Sequence, cast

__all__ = ["Trainer"]


class Trainer:
    """
    Class of the trainer of the model.
    """

    def __init__(self) -> None:
        """
        Constructor of the class.
        """

        self.train_loader, self.test_loader = self._get_loaders()
        self.device = torch.device("cpu")
        self.model = YOLO().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = YOLOLoss()
        self.losses: dict[str, list[float | tuple[int, float]]] = {
            "coord": [],
            "obj": [],
            "noobj": [],
            "class": [],
            "loss": [],
            "map_train": [],
            "map_test": [],
        }
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )

    def _get_loaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Obtains the train and test loaders.

        Returns:
            Train and test loader.
        """

        # TODO
        # Initialize custom dataset objects for training and testing
        train_ds = RaccoonDataset(
            img_dir="data/images_resized/train",
            label_dir="data/labels_resized/train",
        )
        test_ds = RaccoonDataset(
            img_dir="data/images_resized/test",
            label_dir="data/labels_resized/test",
        )

        # Optionally restrict dataset sizes for quicker training/testing
        if len(train_ds) > 160:
            idx: Sequence[int] = list(range(160))
            train_ds = cast(Dataset[Any], Subset(train_ds, idx))
        if len(test_ds) > 32:
            idx2: Sequence[int] = list(range(32))
            test_ds = cast(Dataset[Any], Subset(test_ds, idx2))

        # Create train DataLoader
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

        # Create test DataLoader
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        return train_loader, test_loader

    def _append_losses(
        self,
        coord_loss: torch.Tensor,
        obj_loss: torch.Tensor,
        noobj_loss: torch.Tensor,
        class_loss: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """
        Appends the losses of the batch to the dictionary of the losses.

        Args:
            coord_loss: Loss for the predicted coordinates.
            obj_loss: Loss for the prediction of if an object exists in the cell.
            noobj_loss: Loss for the prediction of if an object does not exist in the
                cell.
            class_loss: Loss for the prediction of the class of the object.
            loss: Total loss.
        """

        self.losses["coord"].append(float(coord_loss))
        self.losses["obj"].append(float(obj_loss))
        self.losses["noobj"].append(float(noobj_loss))
        self.losses["class"].append(float(class_loss))
        self.losses["loss"].append(float(loss))

    def _make_epoch_train(self) -> None:
        """
        Performs one epoch of the training.
        """

        # TODO
        self.model.train()

        # Running sums for per-epoch averages
        sum_coord = 0.0
        sum_obj = 0.0
        sum_noobj = 0.0
        sum_class = 0.0
        sum_total = 0.0
        n_batches = 0

        # Iterate mini-batches
        for images, targets in tqdm(self.train_loader, desc="Training", leave=False):
            param_device = next(self.model.parameters()).device
            images = images.to(param_device, non_blocking=True)
            targets = targets.to(param_device, non_blocking=True)

            # Forward pass -> YOLO grid predictions
            preds = self.model(images)

            # Compute loss components and total
            coord_l, obj_l, noobj_l, class_l, total_l = self.criterion(preds, targets)

            # Optimizer step
            self.optimizer.zero_grad(set_to_none=True)
            total_l.backward()
            self.optimizer.step()

            # Detach -> move to CPU -> convert to float
            sum_coord += float(coord_l.detach().cpu())
            sum_obj += float(obj_l.detach().cpu())
            sum_noobj += float(noobj_l.detach().cpu())
            sum_class += float(class_l.detach().cpu())
            sum_total += float(total_l.detach().cpu())
            n_batches += 1

        # registrar pérdidas promedio por época
        self._append_losses(
            torch.tensor(sum_coord / max(n_batches, 1)),
            torch.tensor(sum_obj / max(n_batches, 1)),
            torch.tensor(sum_noobj / max(n_batches, 1)),
            torch.tensor(sum_class / max(n_batches, 1)),
            torch.tensor(sum_total / max(n_batches, 1)),
        )

    def fit(
        self,
        n_epochs: int = EPOCHS,
        start_epochs: list[int] | None = None,
        save_every: int = 5,
        path_weights: str = PATH_MODEL,
        path_images: str | None = None,
    ) -> None:
        """
        Training of the network. It saves the parameters of the trained model.

        Args:
            n_epochs: Number of epochs to train.
            start_epochs: List with the starting epoch for each graphic saved.
            save_every: To save the model and calculate mAP every 'save_every' epochs.
            path_weights: Path where the weights are saved.
            path_images: Path where we save the images of the evolution of the training.
        """

        # TODO
        for epoch in range(n_epochs):
            tqdm.write(f"Epoch {epoch+1}/{n_epochs}")
            self._make_epoch_train()

            # Periodic checkpoint & metrics
            if (epoch + 1) % save_every == 0 or (epoch + 1) == n_epochs:
                try:
                    torch.save(self.model.state_dict(), path_weights)
                    tqdm.write(f"Saved weights to: {path_weights}")
                except Exception as e:
                    tqdm.write(f"[WARN] Could not save weights: {e}")

                # Compute mAP on train/test if the helper is available
                self.model.eval()
                map_train = 0.0
                map_test = 0.0
                with torch.no_grad():
                    try:
                        map_train = float(
                            mean_average_precision("train", self.model, self.device)
                        )
                        map_test = float(
                            mean_average_precision("test", self.model, self.device)
                        )
                    except TypeError:
                        try:
                            # Intento 2: (model, loader)
                            map_train = float(
                                mean_average_precision("train", self.model)
                            )
                            map_test = float(mean_average_precision("test", self.model))
                        except Exception as e:
                            tqdm.write(f"[WARN] mAP not computed: {e}")
                    except Exception as e:
                        tqdm.write(f"[WARN] mAP not computed: {e}")

                # Store mAP history for plotting later
                self.losses["map_train"].append((epoch + 1, map_train))
                self.losses["map_test"].append((epoch + 1, map_test))
                tqdm.write(f"mAP — train: {map_train:.4f} | test: {map_test:.4f}")

        # Save figures of training dynamics
        if path_images is not None:
            self._plot_training_evolution(
                start_epochs=start_epochs, path_images=path_images
            )

    def test(self) -> float:
        """
        Test of the trained network.

        Returns:
            Loss in the test set.
        """

        # TODO
        self.model.eval()
        sum_total = 0.0
        n_batches = 0

        with torch.no_grad():
            for images, targets in tqdm(self.test_loader, desc="Testing", leave=False):
                # Ensure data is on the same device as the model
                param_device = next(self.model.parameters()).device
                images = images.to(param_device, non_blocking=True)
                targets = targets.to(param_device, non_blocking=True)

                # Forward & compute total loss
                preds = self.model(images)
                _, _, _, _, total_l = self.criterion(preds, targets)

                sum_total += float(total_l.detach().cpu())
                n_batches += 1

        avg = sum_total / max(n_batches, 1)
        tqdm.write(f"Test loss: {avg:.4f}")
        return avg

    def _save_graphic_loss(self, start_epoch: int, path_images: str) -> None:
        """
        Saves a graphic starting from a specific epoch.

        Args:
            start_epoch: First epoch of the graphic.
            path_images: Path where the images are saved.
        """

        x = range(start_epoch + 1, len(self.losses["loss"]) + 1)

        fig, axs = plt.subplots()
        axs.plot(x, self.losses["coord"][start_epoch:], label="Coordinates")
        axs.plot(x, self.losses["obj"][start_epoch:], label="Objects")
        axs.plot(x, self.losses["noobj"][start_epoch:], label="No Objects")
        axs.plot(x, self.losses["class"][start_epoch:], label="Classification")
        axs.plot(x, self.losses["loss"][start_epoch:], label="Total")

        axs.grid()
        axs.legend()
        axs.set_xlabel("Epoch")
        axs.set_ylabel("Loss")
        axs.set_title("Evolution of the Loss During the Training")

        fig.savefig(f"{path_images}_loss_{start_epoch}.png")
        plt.close(fig)

    def _save_graphic_map(self, path_images: str) -> None:
        """
        Saves a graphic starting from a specific epoch.

        Args:
            path_images: Path where the images are saved.
        """

        x = [xx[0] for xx in self.losses["map_train"]]  # type: ignore

        fig, axs = plt.subplots()
        axs.plot(
            x, [xx[1] for xx in self.losses["map_train"]], label="Train"  # type: ignore
        )
        axs.plot(
            x, [xx[1] for xx in self.losses["map_test"]], label="Test"  # type: ignore
        )

        axs.grid()
        axs.legend()
        axs.set_xlabel("Epoch")
        axs.set_ylabel("mAP")
        axs.set_title("Evolution of the mAP During the Training")

        fig.savefig(f"{path_images}_map.png")
        plt.close(fig)

    def _plot_training_evolution(
        self, start_epochs: list[int] | None = None, path_images: str | None = None
    ) -> None:
        """
        Plots the evolution of the training loss.

        Args:
            start_epochs: List with the starting epoch for each graphic saved.
            path_images: Path where the images are saved.
        """

        if start_epochs is None:
            total_epochs = len(self.losses["loss"])
            if total_epochs < 50:
                start_epochs = [0]
            elif total_epochs < 100:
                start_epochs = [0, 50]
            else:
                start_epochs = [0, 50, 100]
        if path_images is None:
            path_images = "images/training/evolution"

        for start_epoch in start_epochs:
            self._save_graphic_loss(start_epoch, path_images=path_images)

        self._save_graphic_map(path_images)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.fit()
