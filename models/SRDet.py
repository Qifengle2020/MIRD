import torch.nn as nn
import matplotlib as plt
from models.MIRNet_model import MIRNet as MIRNet
from models.yolo import Model as yolo5model

class Cnet(nn.Module):
    def __init__(self, cfg='./models/yolov5l.yaml', ch=3, nc=3):
        super(Cnet, self).__init__()
        #scale_factor = 2
        self.sr_layer = MIRNet(in_channels=3, out_channels=3, n_feat=64, kernel_size=3, stride=2, n_RRG=3, n_MSRB=2, height=3, width=2, bias=False)
        self.yolo5 = yolo5model(cfg, ch, nc)
        self.downsample = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        # self.stride = self.yolo5.stride / scale_factor
        self.stride = self.yolo5.stride

    def forward(self, image, augment=False, profile=False):
        out1 = self.downsample(image)
        out1 = self.sr_layer(image)
        out2 = self.yolo5(out1, augment, profile)

        return out2