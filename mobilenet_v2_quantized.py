from camera_interface.OAK_D_API import OAK_D
import cv2

import time
import torch
import numpy as np
from torchvision import models, transforms
import classes
torch.backends.quantized.engine = 'qnnpack'



preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

net = models.quantization.mobilenet_v2(weights=models.quantization.MobileNet_V2_QuantizedWeights, quantize=True)
# jit model to take it from ~20fps to ~30fps
net = torch.jit.script(net)

started = time.time()
last_logged = time.time()
frame_count = 0

# pipeline for interacting with OAK-D camera
oak_d = OAK_D()
with torch.no_grad():
    while True:
        color_frame = oak_d.get_color_frame(show_fps=True)
        # cv2.imshow("frame", color_frame)

        # preprocess
        input_tensor = preprocess(color_frame)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        # run model
        output = net(input_batch)

        # print top 3 classes predicted by model
        # top = list(enumerate(output[0].softmax(dim=0)))
        # top.sort(key=lambda x: x[1], reverse=True)
        # for idx, val in top[:3]:
        #     print(f"{val.item()*100:.2f}% {classes.image_net_classes[idx]}")
        # print('---------------------------')

        # log model performance
        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            print(f"{frame_count / (now-last_logged)} fps")
            last_logged = now
            frame_count = 0
            
            if cv2.waitKey(1) == ord('q'):
                break