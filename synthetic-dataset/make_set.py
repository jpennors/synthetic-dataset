import cv2
from utils import read_video
from image_blending import basic_blending, seamless_clone_blending
import numpy as np
import os
import random
from functools import partial

BLENDING_METHODS = {
    'basic_blending' : basic_blending,
    'seamless_clone_mixed' : partial(seamless_clone_blending, clone_type=cv2.MIXED_CLONE),
    'seamless_clone_normal' : partial(seamless_clone_blending, clone_type=cv2.MIXED_CLONE),
    'seamless_clone_monochrome' : partial(seamless_clone_blending, clone_type=cv2.MONOCHROME_TRANSFER)
}

DATASET_BASE_PATH = 'dataset'

def save_img(folder_path, filename, img):
    os.makedirs(folder_path, exist_ok=True)
    cv2.imwrite(f'{folder_path}/{filename}', img)

def make_one_set(smoke_video_file, background_file, set_idx, fx=0.3, 
                 fy=0.2, opacity=0.8, smoke_speed=5, smoke_offset=20):

    # Get smokes frames
    smoke_imgs = read_video(smoke_video_file)

    # Resize smokes frames
    smoke_imgs = [cv2.resize(smoke_img, (0, 0), fx=fx, fy=fy) for smoke_img in smoke_imgs[smoke_offset::smoke_speed]]
    # Compute mask
    smoke_mask = 0*smoke_imgs[0]
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
    imgs = read_video(background_file)

    name = 'set_' + str(set_idx).zfill(3) + '_'

    # Random offset
    hs, ws = smoke_imgs[0].shape[:2]
    hbg, wbg = imgs[0].shape[:2]

    if hs < hbg and ws < wbg:
        dy = random.randint(0, hbg - hs - 1)
        dx = random.randint(0, wbg - ws - 1)

        for blending_type, blending_method in BLENDING_METHODS.items():
            
            print(f'Starting {blending_type} ...')
            
            for i, (img, smoke) in enumerate(zip(imgs, smoke_imgs)):
                result, mask = blending_method(img, smoke, offset=(dy, dx))
                
                save_img(f'{DATASET_BASE_PATH}/{blending_type}/img', name + str(i).zfill(4) + '.png', result)
                save_img(f'{DATASET_BASE_PATH}/{blending_type}/mask', name + str(i).zfill(4) + '.jpg', mask*255)
                save_img(f'{DATASET_BASE_PATH}/{blending_type}/smoke', name + str(i).zfill(4) + '.jpg', smoke)
