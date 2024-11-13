import cv2
import numpy as np
import os

from PIL import Image
from glob import glob

OUT_DIR = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/data/bootstrapped-dataset-4x4"
TARGET_IMG_SIDE_LEN = 512
NUM_SLICES = 2

# 1. open image, subdivide into 64x64 squares
# 2. upscale img_chunk -> (512, 512)
# 3. save


def bootstrap_img(fp):
    img = cv2.imread(fp)
    img_name = os.path.basename(fp)
    dst_folder = OUT_DIR
    if "target" in fp:
        dst_subdir = "target"
    else:
        dst_subdir = "source"
    dst_subdir_path = os.path.join(dst_folder, dst_subdir)
    os.makedirs(dst_subdir_path, exist_ok=True)
    slice_size = img.shape[0] // NUM_SLICES
    idx = 0
    for i in range(NUM_SLICES):
        for j in range(NUM_SLICES):
            h = slice_size * i
            w = slice_size * j
            arr_seg = img[h : h + slice_size, w : w + slice_size, :]
            img_seg = Image.fromarray(arr_seg)
            img_seg = img_seg.resize((TARGET_IMG_SIDE_LEN, TARGET_IMG_SIDE_LEN))
            img_name_without_ext = os.path.splitext(img_name)[0]
            dst_fp = os.path.join(
                dst_folder, dst_subdir, f"{img_name_without_ext}_{idx}.png"
            )
            img_seg.save(dst_fp)
            idx += 1

if __name__ == "__main__":
    fp = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/data/full-sized-toy-dataset"
    imgs = glob(os.path.join(fp, "*", "*.png"))
    for img_fp in imgs:
        bootstrap_img(img_fp)
