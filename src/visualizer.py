# src/visualizer.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader


from src.data_manager import DataManager
from src.model_handler import ModelHandler
from src.strategies import FeatureExtractionStrategy


class FeatureVisualizer:
    """
    A class to visualize high-dimensional features using PCA.
    """

    def __init__(self, config: dict) -> None:
        """
        Initializes the FeatureVisualizer with a configuration.

        Args:
            config (dict): A configuration dictionary containing data, model,
            and training settings.
        """
        self.config = config
        self.data_manager = DataManager(config)
        self.model_handler = ModelHandler(config)

    def visualize_features(self) -> None:
        """
        Runs the feature extraction, PCA, and visualization pipeline.
        """
        # TODO
        # Load only the training split
        self.data_manager.load_train_data()
        train_loader = self.data_manager.get_train_loader()

        self.model_handler.load_model()

        strategy = FeatureExtractionStrategy(self.model_handler)

        # Backward-compatible hook names
        if hasattr(strategy, "prepare"):
            strategy.prepare()
        elif hasattr(strategy, "setup"):
            strategy.setup()

        model = self.model_handler.model
        model.eval()
        model.to("cpu")

        # Extract features and produce the PCA plot
        self._extract_features_and_plot(train_loader)

    def _extract_features_and_plot(self, train_loader: DataLoader) -> None:
        """
        Extract features and calls _plot_pca to represent them.

        Args:
            features_2d (np.ndarray): The 2D array of features after PCA.
            labels (np.ndarray): The corresponding labels for the features.
        """

        # TODO
        model = self.model_handler.model
        assert model is not None, "Model must be loaded before extracting features."

        max_samples = int(self.config.get("visualizer", {}).get("max_samples", 512))
        features_list = []
        labels_list = []
        collected = 0

        # Disable gradients to speed up inference and reduce memory
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to("cpu")

                outputs = model(images)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                feats = outputs.detach().cpu().numpy()
                features_list.append(feats)
                labels_list.append(labels.detach().cpu().numpy())

                collected += images.size(0)
                if collected >= max_samples:
                    break

        # Concatenate and truncate to the requested max_samples
        features = np.concatenate(features_list, axis=0)[:max_samples]
        labels = np.concatenate(labels_list, axis=0)[:max_samples]

        # PCA to 2D for visualization
        pca = PCA(n_components=2, random_state=0)
        features_2d = pca.fit_transform(features)

        self._plot_pca(features_2d, labels)

    def _plot_pca(self, features_2d: np.ndarray, labels: np.ndarray) -> None:
        """
        Plots the 2D PCA-transformed features.

        Args:
            features_2d (np.ndarray): The 2D array of features after PCA.
            labels (np.ndarray): The corresponding labels for the features.
        """
        # TODO
        plt.figure(figsize=(6, 5))
        # Color points by label
        _ = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, s=12, alpha=0.8)
        plt.title("PCA of extracted features")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")

        # Optional save to file if a path is provided in the config
        save_path = self.config.get("visualizer", {}).get("save_path")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        plt.show()
        plt.close()
