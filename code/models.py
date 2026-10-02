"""
Definition of the architecture of the several CNN models employed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


N_OUTPUTS = 12


class CustomCNN(nn.Module):
    """Custom CNN"""

    def __init__(self, n_outputs=N_OUTPUTS):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.InstanceNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.InstanceNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.InstanceNorm2d(64)

        self.conv4 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.bn4 = nn.InstanceNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.ln1 = nn.LayerNorm(512)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(512, n_outputs)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)

        x = x.view(-1,  32 * 8 * 8)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout(x)

        return self.fc2(x)


def build_model(name, n_outputs=N_OUTPUTS):
    """Build the selected architecture"""
    if name == "Custom CNN":
        return CustomCNN(n_outputs)

    if name == "ResNet18":
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, n_outputs),
        )
        return model

    if name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=None)
        model.features[0][0] = nn.Conv2d(1, 32, 3, 2, 1, bias=False)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, n_outputs),
        )
        return model

    if name == "MobileNetV3":
        model = models.mobilenet_v3_small(weights=None)
        model.features[0][0] = nn.Conv2d(1, 16, 3, 2, 1, bias=False)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, n_outputs),
        )
        return model

    if name == "VGG16":
        model = models.vgg16(weights=None)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        model.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        model.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, n_outputs),
        )
        return model

    raise ValueError(
        f"Unknown model '{name}'. "
        f"Available models: {', '.join(available_models())}"
    )


def available_models():
    return [
        "Custom CNN",
        "ResNet18",
        "EfficientNetB0",
        "MobileNetV3",
        "VGG16",
    ]
