import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

# Usage of ResNet50 for the encoder.
class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        # Represents each layer from the ResNet to use.
        self.f = []
        
        for name, module in resnet50().named_children():
            # Modify the first convolutional layer.
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # If the layer is a linear layer, or a maxpool layer, DO NOT append 
            # into self.f.
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        # passing of the input into the encoder, followed by the computation of 
        # the projection head, which produces an output of the shape
        # (batch_dim, feature_dim), followed by its normalization.
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
