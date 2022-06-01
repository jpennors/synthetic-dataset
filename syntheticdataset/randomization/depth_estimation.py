# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import torch
import numpy as np
import skimage as sk

# Depth Deep Learning Model
MIDAS_LARGE = (
    "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
)
MIDAS_HYBRID = (
    "DPT_Hybrid"  # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
)
MIDAS_SMALL = (
    "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
)


class DepthEstimation:
    def __init__(self, model_type=MIDAS_LARGE):

        # Preloading model
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.midas.to(self.device)
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        self.transform = (
            midas_transforms.dpt_transform
            if model_type == MIDAS_LARGE or model_type == MIDAS_HYBRID
            else midas_transforms.small_transform
        )

    def _scale_output(self, output):
        """
        Scaling Depth Output Image
        This methods relies on empirical tests
        The maximum value of a single pixel is 40 and it needs to
        to be scaled between 0 and 255

        Args:
            output (np.array): background image

        Returns:
            np.array: result scaled image
        """
        return (output / 40 * 255).astype(int)

    def estimate_depth_from_image(self, img):
        """
        Estimate the depth the image pixel by pixel with Torch

        Args:
            img (np.array): background image

        Returns:
            np.array: result depth image
        """

        input = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()

        return self._scale_output(output)

    def clean_mask_with_labelization(self, mask):
        """
        This method is cleaning the generated mask.
        Some noise can on the image such as written text at the top corner.
        Every different detected area will be split and the biggest one will be
        kept and considered as sky.

        Args:
            masj (np.array): mask image

        Returns:
            np.array cleaned mask
        """

        blobs_labels = sk.measure.label(mask, background=0)
        s = [np.sum(blobs_labels == l) for l in np.unique(blobs_labels)[1:]]

        if len(s) == 0:
            return mask

        mask = blobs_labels == (np.where(s == np.max(s))[0] + 1)

        return mask

    def generate_mask(self, depth_image, min_threshold, max_threshold):
        """
        Generate a mask without the sky
        This methods relies on empirical tests

        Args:
            img (np.array): background image
            min_threshold (int): min pixel value not to be considered as sky
            max_threshold (int): max pixel value not to be considered as sky

        Returns:
            np.array: result mask image
        """

        depth_image[
            np.logical_or(depth_image < min_threshold, depth_image > max_threshold)
        ] = 0
        depth_image[
            np.logical_and(depth_image >= min_threshold, depth_image <= max_threshold)
        ] = 255

        return self.clean_mask_with_labelization(depth_image)

    def detect_sky_from_depth(self, img, min_threshold=60, max_threshold=170):
        """
        Method to estimate the depth from an image
        and automatically generate the mask from it
        to remove the sky pixels

        Args:
            img (np.array): background image
            min_threshold (int): min pixel value not to be considered as sky
            max_threshold (int): max pixel value not to be considered as sky

        Returns:
            np.array: result mask image
        """

        depth_image = self.estimate_depth_from_image(img)

        return self.generate_mask(depth_image, min_threshold, max_threshold)
