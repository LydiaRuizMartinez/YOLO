"""
This script transform original images and labels into the correct format. In particular,
it reshapes all images into the same size and transforms the corresponding labels. Each
label will be a txt file with as many lines as bounding boxes there are in the image.
"""

import os
import shutil
from typing import Literal

import polars as pl
from PIL import Image

from src.constants import TARGET_SIZE_IMG


def delete_all_dirs() -> None:
    """
    Deletes all created directories if they existed to avoid problems.
    """

    shutil.rmtree("data/images", ignore_errors=True)
    shutil.rmtree("data/images_resized", ignore_errors=True)
    shutil.rmtree("data/labels", ignore_errors=True)
    shutil.rmtree("data/labels_resized", ignore_errors=True)


def _update_original_data(subset: Literal["train", "test"]) -> None:
    """
    Updates the original data for a specific subset.

    Args:
        subset: Subset where to apply the function.
    """

    # Load df
    df = pl.read_csv(f"data/labels_old/{subset}_labels.csv")

    # Order by idx
    df = df.with_columns(
        df["filename"].str.extract(r"-(\d+)\.").cast(pl.Int32).alias("idx") - 1
    ).sort(by="idx")

    # Reset idx to start from 0
    df = df.with_columns((pl.col("idx").rank("dense") - 1).cast(pl.Int32).alias("idx"))

    # Update filename
    df = df.with_columns(
        pl.concat_str(
            [pl.lit("raccoon-"), df["idx"].cast(pl.Utf8), pl.lit(".jpg")]
        ).alias("new_filename")
    )

    # Save df
    path = "data/labels"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    df.write_csv(f"{path}/{subset}_labels.csv")

    # Save images
    path = f"data/images/{subset}"
    os.makedirs(path, exist_ok=True)
    for filename, new_filename in zip(
        df["filename"].to_list(), df["new_filename"].to_list()
    ):
        img = Image.open(f"data/images_old/{filename}")
        img.save(f"data/images/{subset}/{new_filename}")


def update_original_data() -> None:
    """
    Divides the original folder of jpg images into train and test. It also resets index
    in these jpg images and in the csvs.
    """

    for subset in ["train", "test"]:
        _update_original_data(subset)  # type: ignore


def _transform_images_subset(mode: Literal["train", "test"]) -> None:
    """
    Creates a new folder for resized images of train or test and resizes them.

    Args:
        target_size: New size for the images.
        mode: To resize train or test images.
    """

    base_path = "data/images"
    images_path = f"{base_path}/{mode}"
    output_path = f"{base_path}_resized/{mode}"

    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(images_path):
        img = Image.open(f"{images_path}/{filename}")
        img_resized = img.resize(TARGET_SIZE_IMG)
        img_resized.save(f"{output_path}/{filename}")


def transform_images() -> None:
    """
    Creates a new folder for resized images and resizes them.
    """

    for mode in ["train", "test"]:
        _transform_images_subset(mode)  # type: ignore


def transform_labels() -> None:
    """
    Creates a new folder for the new labels and they are saved as txt files. They will
    be normalized and in the format (class, center_x, center_y, width, height).
    """

    labels_path = "data/labels"
    output_path_base = f"{labels_path}_resized"
    mapping_classes = {"raccoon": 0}

    os.makedirs(output_path_base, exist_ok=True)

    for subset in ["train", "test"]:
        df = pl.read_csv(f"{labels_path}/{subset}_labels.csv")
        df = df.with_columns(
            [
                (((pl.col("xmin") + pl.col("xmax")) / 2) / pl.col("width")).alias(
                    "center_x"
                ),
                (((pl.col("ymin") + pl.col("ymax")) / 2) / pl.col("height")).alias(
                    "center_y"
                ),
                ((pl.col("xmax") - pl.col("xmin")) / pl.col("width")).alias(
                    "bbox_width"
                ),
                ((pl.col("ymax") - pl.col("ymin")) / pl.col("height")).alias(
                    "bbox_height"
                ),
                (pl.col("new_filename").str.extract(r"(\d+)").alias("idx")),
                (pl.col("class").replace(mapping_classes).alias("class_id")),
            ]
        )

        output_path = f"{output_path_base}/{subset}"
        os.makedirs(output_path, exist_ok=True)
        for i, idx in enumerate(df["idx"].to_list()):
            filename = f"{output_path}/{idx}.txt"
            if os.path.exists(filename):
                mode = "a"
            else:
                mode = "w"
            with open(filename, mode, encoding="utf8") as f:
                label = (
                    f"{df['class_id'][i]} {df['center_x'][i]} {df['center_y'][i]} "
                    f"{df['bbox_width'][i]} {df['bbox_height'][i]}\n"
                )
                f.write("".join(label))


def main() -> None:
    """
    Deletes all created directories if they existed, transforms the original label dfs
    to the correct format and generates the corresponding images and labels.
    """

    delete_all_dirs()
    update_original_data()
    transform_images()
    transform_labels()


if __name__ == "__main__":
    main()
