import torch.nn as nn
from torchvision.models import ResNet50_Weights, Swin_T_Weights, resnet50, swin_t


class SLFModel(nn.Module):
    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        if pretrained_backbone:
            print('!init with pretrained weights!')
        else:
            print('!random init!')

        swin_weights = Swin_T_Weights.DEFAULT if pretrained_backbone else None
        res_weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None

        self.backbone_1 = swin_t(weights=swin_weights)
        self.backbone_1.head = nn.Linear(768, 2, bias=True)

        self.backbone_2 = resnet50(weights=res_weights)
        self.backbone_2.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2),
        )

        self.backbone_3 = resnet50(weights=res_weights)
        self.backbone_3.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2),
        )

    def forward(self, x_1, x_2, x_3):
        z_1 = self.backbone_1(x_1)
        z_2 = self.backbone_2(x_2)
        z_3 = self.backbone_3(x_3)
        return z_1, z_2, z_3
