from dataclasses import dataclass
from typing import List, Tuple
from torchvision.ops import masks_to_boxes
from scipy.ndimage import label as scipy_label
import numpy as np
import torch

import matplotlib.pyplot as plt

import matplotlib.patches as patches

bbox = Tuple[float, float, float, float]


@dataclass
class LiverSlice:
    liver_image: str
    label_image: str
    slice_number: int
    has_liver: bool
    has_cancer: bool
    cancer_bbox: List[bbox]


def plot_bbs_on_slice(
    ax: plt.Axes, image_slice: np.ndarray, bounding_boxes: List[bbox]
):
    """Plots the bounding boxes on the image slice

    Args:
        ax (plt.Axes): The axes to plot on
        image_slice (np.ndarray): The image slice to plot
        bounding_boxes (List[bbox]): The bounding boxes to plot
    """
    ax.imshow(image_slice, cmap="bone")

    # Assuming bounding boxes are in (xmin, ymin, xmax, ymax) format
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
