from dataclasses import dataclass
from typing import List, Tuple

import nibabel
from torchvision.ops import masks_to_boxes
from scipy.ndimage import label as scipy_label
import numpy as np
import torch

import matplotlib.pyplot as plt

import matplotlib.patches as patches

bbox = Tuple[float, float, float, float]

LABEL_BACKGROUND = 0
LABEL_LIVER = 1
LABEL_CANCER = 2
LIVER_DATA_PATH = "./Task03_Liver"


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


def get_image_data(path_in_dir, dir=LIVER_DATA_PATH) -> np.ndarray:
    image = nibabel.load(f"{dir}/{path_in_dir}")
    return image.get_fdata()


def get_image_slice_data(path_in_dir, slice_idx, dir=LIVER_DATA_PATH) -> np.ndarray:
    image = get_image_data(path_in_dir, dir)
    return image[:, :, slice_idx]


def get_cancer_bounding_boxes(segmented_liver: np.ndarray) -> List[bbox]:
    """
    Returns a list of bounding boxes for the cancerous regions in the image.

    Args:
      segmented_liver: A 2D numpy array with the segmented liver

    Returns:
      A list of tuples with the bounding boxes of the cancer in the image, (xmin, ymin, xmax, ymax) format
    """
    cancer_mask = segmented_liver == LABEL_CANCER
    if cancer_mask.sum() == 0:
        # no cancer
        return []

    labeled_mask, num_labels = scipy_label(cancer_mask)
    bounding_boxes = []
    for i in range(1, num_labels + 1):
        mask_i = labeled_mask == i
        mask_i = torch.from_numpy(mask_i)
        box = masks_to_boxes(
            mask_i.unsqueeze(dim=0)
        )  # unsqueeze to add an extra dimension at 0th index
        bounding_boxes.append(
            (box[0][0].item(), box[0][1].item(), box[0][2].item(), box[0][3].item())
        )

    return bounding_boxes
