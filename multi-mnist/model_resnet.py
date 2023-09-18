# resnet18 base model for Pareto MTL
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class ResnetModel(torch.nn.Module):
    def __init__(self, n_tasks):
        super(ResnetModel, self).__init__()
        self.n_tasks = n_tasks
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        fc_in_features = self.encoder.fc.in_features
        self.encoder.fc = torch.nn.Linear(fc_in_features, 100)

        for i in range(self.n_tasks):
            setattr(self, "task_{}".format(i + 1), nn.Linear(100, 10))

    def forward(self, x):
        x = F.relu(self.encoder(x))
        outs = []
        for i in range(self.n_tasks):
            layer = getattr(self, "task_{}".format(i + 1))
            outs.append(layer(x))

        return outs

    def get_shared_parameters(self):
        params = [
            {"params": self.encoder.parameters(), "lr_mult": 1},
        ]
        return params

    def get_classifier_parameters(self):
        params = []
        for i in range(self.n_tasks):
            layer = getattr(self, "task_{}".format(i + 1))
            params.append({"params": layer.parameters(), "lr_mult": 1})
        return params
