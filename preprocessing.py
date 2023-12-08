import os
from pathlib import Path

from PIL import Image


def select_with_mask(pic_path, mask_path, out_dir):
    img = Image.open(pic_path)
    im_size = img.size
    mask = Image.open(mask_path)
    plane = Image.new("RGBA", img.size, (0, 0, 0, 0))
    res = Image.composite(img, plane, mask)
    res = res.crop(res.getbbox())
    plane = Image.new("RGB", im_size, (0, 0, 0, 0))
    plane.paste(res, ((im_size[0] - res.size[0]) // 2, (im_size[1] - res.size[1]) // 2))
    plane.save(out_dir / pic_path.name)


def preprocess_images():
    dataset_path = Path("./dataset/")
    in_path = dataset_path / "data"
    out_path = dataset_path / "preprocessed_data"

    in_train_path = in_path / "train_images"
    in_train_masks_path = in_path / "train_lung_masks"
    in_test_path = in_path / "test_images"
    in_test_masks_path = out_path / "test_images_masks"

    out_train_path = out_path / "train_images"
    out_test_path = out_path / "test_images"

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(out_train_path):
        os.mkdir(out_train_path)
    if not os.path.exists(out_test_path):
        os.mkdir(out_test_path)
    
    # processing train images
    for filename in os.listdir(in_train_path):
        select_with_mask(
            in_train_path / filename,
            in_train_masks_path / filename,
            out_train_path
        )

    # processing test images
    for filename in os.listdir(in_test_path):
        select_with_mask(
            in_test_path / filename,
            in_test_masks_path / filename,
            out_test_path
        )


if __name__ == "__main__":
    preprocess_images()
