import torch
from ultralytics import YOLO


print(torch.cuda.is_available())

model = YOLO("best (3).pt")

result=model.track(source="A Eagle Flying.mp4",show=True,tracker="bytetrack.yaml",conf=0.3, iou=0.5)
