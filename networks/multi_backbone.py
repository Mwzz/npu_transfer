from torchvision.models import resnet50, resnet101, swin_t
import torch.nn as nn


def _build_resnet50(pretrained: bool = False):
    try:
        return resnet50(weights=None if not pretrained else "DEFAULT")
    except TypeError:
        return resnet50(pretrained=pretrained)


def _build_resnet101(pretrained: bool = False):
    try:
        return resnet101(weights=None if not pretrained else "DEFAULT")
    except TypeError:
        return resnet101(pretrained=pretrained)


def _build_swin_t(pretrained: bool = False):
    try:
        return swin_t(weights=None if not pretrained else "DEFAULT")
    except TypeError:
        return swin_t(pretrained=pretrained)


class Multi_backbone_ResNet(nn.Module):
    def __init__(self, arch):
        super().__init__()
        if arch == "res50":
            self.backbone_1 = _build_resnet50(pretrained=False)
            self.backbone_1.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
            self.backbone_2 = _build_resnet50(pretrained=False)
            self.backbone_2.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
            self.backbone_3 = _build_resnet50(pretrained=False)
            self.backbone_3.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
        elif arch == "res101":
            self.backbone_1 = _build_resnet101(pretrained=False)
            self.backbone_1.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
            self.backbone_2 = _build_resnet101(pretrained=False)
            self.backbone_2.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
            self.backbone_3 = _build_resnet101(pretrained=False)
            self.backbone_3.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
        elif arch == "Swin_T":
            self.backbone_1 = _build_swin_t(pretrained=False)
            self.backbone_1.head = nn.Linear(768, 2, bias=True)
            self.backbone_2 = _build_swin_t(pretrained=False)
            self.backbone_2.head = nn.Linear(768, 2, bias=True)
            self.backbone_3 = _build_swin_t(pretrained=False)
            self.backbone_3.head = nn.Linear(768, 2, bias=True)
        elif arch == "Swin_T_mix":
            self.backbone_1 = _build_swin_t(pretrained=False)
            self.backbone_1.head = nn.Linear(768, 2, bias=True)
            self.backbone_2 = _build_swin_t(pretrained=False)
            self.backbone_2.head = nn.Linear(768, 2, bias=True)
            self.backbone_3 = _build_resnet50(pretrained=False)
            self.backbone_3.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
        else:
            raise ValueError(f"Unsupported arch: {arch}")

    def forward(self, x_1, x_2, x_3):
        x_1 = self.backbone_1(x_1)
        x_2 = self.backbone_2(x_2)
        x_3 = self.backbone_3(x_3)
        return x_1, x_2, x_3


class Multi_backbone_ResNet_v2(nn.Module):
    def __init__(self, arch):
        super().__init__()
        if arch == "res50":
            self.backbone_1 = _build_resnet50(pretrained=False)
            self.backbone_1.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
            self.backbone_2 = _build_resnet50(pretrained=False)
            self.backbone_2.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
            self.backbone_3 = _build_resnet50(pretrained=False)
            self.backbone_3.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
        elif arch == "res101":
            self.backbone_1 = _build_resnet101(pretrained=False)
            self.backbone_1.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
            self.backbone_2 = _build_resnet101(pretrained=False)
            self.backbone_2.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
            self.backbone_3 = _build_resnet101(pretrained=False)
            self.backbone_3.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
        elif arch == "Swin_T":
            self.backbone_1 = _build_swin_t(pretrained=False)
            self.backbone_1.head = nn.Linear(768, 2, bias=True)
            self.backbone_2 = _build_swin_t(pretrained=False)
            self.backbone_2.head = nn.Linear(768, 2, bias=True)
            self.backbone_3 = _build_swin_t(pretrained=False)
            self.backbone_3.head = nn.Linear(768, 2, bias=True)
        elif arch == "Swin_T_mix":
            self.backbone_1 = _build_swin_t(pretrained=False)
            self.backbone_1.head = nn.Linear(768, 2, bias=True)
            self.backbone_2 = _build_swin_t(pretrained=False)
            self.backbone_2.head = nn.Linear(768, 2, bias=True)
            self.backbone_3 = _build_resnet50(pretrained=False)
            self.backbone_3.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
        elif arch == "Swin_T_mix_TFS":
            print("Init Swin_T_mix_TFS")
            self.backbone_1 = _build_swin_t(pretrained=False)
            self.backbone_1.head = nn.Linear(768, 2, bias=True)
            self.backbone_2 = _build_swin_t(pretrained=False)
            self.backbone_2.head = nn.Linear(768, 2, bias=True)
            self.backbone_3 = _build_resnet50(pretrained=False)
            self.backbone_3.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
        else:
            raise ValueError(f"Unsupported arch: {arch}")

    def forward(self, x_1, x_2_1, x_2_2, x_3):
        x_1 = self.backbone_1(x_1)
        x_2_1 = self.backbone_2(x_2_1)
        x_2_2 = self.backbone_2(x_2_2)
        x_3 = self.backbone_3(x_3)
        return x_1, x_2_1, x_2_2, x_3


class Multi_backbone_ResNet_v2_part(nn.Module):
    def __init__(self, arch):
        super().__init__()
        if arch == "res50":
            self.backbone_1 = _build_resnet50(pretrained=False)
            self.backbone_1.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
            self.backbone_2 = _build_resnet50(pretrained=False)
            self.backbone_2.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
            self.backbone_3 = _build_resnet50(pretrained=False)
            self.backbone_3.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
        elif arch == "res101":
            self.backbone_1 = _build_resnet101(pretrained=False)
            self.backbone_1.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
            self.backbone_2 = _build_resnet101(pretrained=False)
            self.backbone_2.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
            self.backbone_3 = _build_resnet101(pretrained=False)
            self.backbone_3.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
        elif arch == "Swin_T":
            self.backbone_1 = _build_swin_t(pretrained=False)
            self.backbone_1.head = nn.Linear(768, 2, bias=True)
            self.backbone_2 = _build_swin_t(pretrained=False)
            self.backbone_2.head = nn.Linear(768, 2, bias=True)
            self.backbone_3 = _build_swin_t(pretrained=False)
            self.backbone_3.head = nn.Linear(768, 2, bias=True)
        elif arch == "Swin_T_mix":
            self.backbone_1 = _build_swin_t(pretrained=False)
            self.backbone_1.head = nn.Linear(768, 2, bias=True)
            self.backbone_2 = _build_swin_t(pretrained=False)
            self.backbone_2.head = nn.Linear(768, 2, bias=True)
            self.backbone_3 = _build_resnet50(pretrained=False)
            self.backbone_3.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
        elif arch == "Swin_T_mix_TFS":
            print("Init Swin_T_mix_TFS")
            self.backbone_1 = _build_swin_t(pretrained=False)
            self.backbone_1.head = nn.Linear(768, 2, bias=True)
            self.backbone_2 = _build_swin_t(pretrained=False)
            self.backbone_2.head = nn.Linear(768, 2, bias=True)
            self.backbone_3 = _build_resnet50(pretrained=False)
            self.backbone_3.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2),
            )
        else:
            raise ValueError(f"Unsupported arch: {arch}")

    def forward(self, x_1, x_3):
        x_1 = self.backbone_1(x_1)
        x_3 = self.backbone_3(x_3)
        return x_1, x_3
