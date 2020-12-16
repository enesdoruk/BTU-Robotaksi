import torch
from PIL import Image
import cv2
import os

model = torch.hub.load('/yolov5', 'yolov5s', path_or_model='best.pt', pretrained=False)
model = model.autoshape()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

cap = cv2.VideoCapture("project.avi")

while(True):

    ret, frame = cap.read()

    frame = cv2.resize(frame, (416,416))
    
    prediction = model(frame, size=416)

    cv2.imshow('frame',prediction)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
