import os
from pathlib import Path
import shutil

from PIL import Image
import cv2
import torch
import torchvision


def build_masks():
    exceptions_list = os.listdir("./exceptions")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=False)
    unet = unet.to(device)
    m_state_dict = torch.load("./unet_dump.pt", map_location=device)
    unet.load_state_dict(m_state_dict)

    dataset_path = Path("./dataset/")
    in_test_path = dataset_path / "data" / "test_images"
    out_path = dataset_path / "preprocessed_data"
    out_test_masks_path = out_path / "test_images_masks"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(out_test_masks_path):
        os.mkdir(out_test_masks_path)

    transform_to_img = torchvision.transforms.ToPILImage()
    transform_to_tensor = torchvision.transforms.ToTensor()
    for filename in os.listdir(in_test_path):
        if filename not in exceptions_list:
            img = cv2.imread(str(in_test_path / filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform_to_tensor(img).to(device)
            pred = unet(tensor.unsqueeze(0)).squeeze(0)
            mask = transform_to_img(pred)
            mask.save(out_test_masks_path / filename)
        else:
            shutil.copy(f"./exceptions/{filename}", str(out_test_masks_path / filename))


if __name__ == "__main__":
    build_masks()
