import csv
import os
from pathlib import Path

import cv2
import torch
import torchvision
import torch.nn as nn
import tqdm



def predict_classes():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(in_features=2048, out_features=3, bias=True)
    model = model.to(device)
    m_state_dict = torch.load("./predict_dump.pt", map_location=device)
    model.load_state_dict(m_state_dict)
    model.eval()

    res = []
    test_path = Path("./dataset/preprocessed_data/test_images")
    transform_to_tensor = torchvision.transforms.ToTensor()
    for i, filename in tqdm.tqdm(enumerate(os.listdir(test_path))):
        img = cv2.imread(str(test_path / filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tensor = transform_to_tensor(img).unsqueeze(0).to(device)
        ans = torch.argmax(model(tensor))
        res.append([i, int(ans)])
    return res
        

if __name__ == "__main__":
    res = predict_classes()
    with open("predictions.csv", "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", "target_feature"])
        writer.writerows(res)
