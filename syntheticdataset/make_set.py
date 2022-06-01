# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import cv2
import numpy as np
import random
from syntheticdataset.utils import read_video, save_img, save_label, get_label
from syntheticdataset.image_blending import basic_blending, poisson_blending
from syntheticdataset.randomization.randomization import Randomization


BLENDING_METHODS = {
    "basic_blending": basic_blending,
    "poisson_blending": poisson_blending,
}


def make_one_set(
    smoke_video_file,
    background_file,
    root="pyro_dataset",
    set_idx=0,
    fx=0.3,
    fy=0.2,
    opacity=0.8,
    smoke_speed=5,
    smoke_offset=20,
    train=True,
    save_mask=False,
    save_bbox=False,
    size_max_bg=1280,
    size_max_smoke=1280,
):

    randomization = Randomization()

    # Get smokes frames
    smoke_imgs = read_video(smoke_video_file, size_max=size_max_smoke)

    # Resize smokes frames
    smoke_imgs = [
        cv2.resize(smoke_img, (0, 0), fx=fx, fy=fy)
        for smoke_img in smoke_imgs[smoke_offset::smoke_speed]
    ]
    # Compute mask
    smoke_mask = 0 * smoke_imgs[0]
    for smoke_img in smoke_imgs:
        smoke_mask[smoke_img > 50] = 255

    y, x = np.where(smoke_mask[:, :, 0] == 255)
    x0 = min(x)
    x1 = max(x)
    y0 = min(y)
    y1 = max(y)

    # Apply mask
    smoke_imgs = [smoke_img[y0:y1, x0:x1, :] for smoke_img in smoke_imgs]

    # Read background
    imgs = read_video(background_file, size_max=size_max_bg)

    name = "set_" + str(set_idx).zfill(3) + "_"

    # Random offset
    hs, ws = smoke_imgs[0].shape[:2]
    hbg, wbg = imgs[0].shape[:2]
    dx, dy = randomization.get_random_start_point(imgs[0])

    if hs < hbg and ws < wbg and dy > 0:

        train_val = "train" if train else "val"

        for blending_type, blending_method in BLENDING_METHODS.items():

            for i, (img, smoke) in enumerate(zip(imgs, smoke_imgs)):

                result, mask = blending_method(img, smoke, offset=(dy, dx))

                label = get_label(mask * 255)

                save_img(
                    f"{root}/images/{train_val}/",
                    blending_type + "_" + name + str(i).zfill(4) + ".png",
                    result,
                )
                if save_bbox:
                    save_label(
                        f"{root}/labels/{train_val}/",
                        blending_type + "_" + name + str(i).zfill(4) + ".txt",
                        label,
                    )
                if save_mask:
                    save_img(
                        f"{root}/mask/{train_val}/",
                        blending_type + "_" + name + str(i).zfill(4) + ".jpg",
                        mask * 255,
                    )
