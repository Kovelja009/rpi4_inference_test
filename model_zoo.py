import torch
from torchvision import models
from torchvision.models.detection.anchor_utils import AnchorGenerator
from ultralytics import YOLO
torch.backends.quantized.engine = 'qnnpack'


def mobilenet_v2_quantized():
    net = models.quantization.mobilenet_v2(weights=models.quantization.MobileNet_V2_QuantizedWeights.DEFAULT, quantize=True)
    # jit model to take it from ~20fps to ~30fps
    net = torch.jit.script(net)
    return net

def mobilenet_v3_large_quantized():
    net = models.quantization.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT, quantized=True)
    # jit model to take it from ~20fps to ~30fps
    net = torch.jit.script(net)
    return net

def yolo_v8n():
    net = YOLO('yolov8n.pt')
    return net


model_zoo = {
    "yolo_v8n": yolo_v8n(),
    "mobilenet_v3_large_quantized": mobilenet_v3_large_quantized(),
    "mobilenet_v2_quantized": mobilenet_v2_quantized(),
} 